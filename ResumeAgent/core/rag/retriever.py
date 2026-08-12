"""
core/rag/retriever.py - 增强混合检索器

实现完整的增强RAG检索链路:
1. BM25 关键词检索（稀疏检索）
2. Chroma 向量相似度检索（稠密检索）
3. RRF (Reciprocal Rank Fusion) 融合结果
4. CrossEncoder 重排序 + 低质量过滤

对外统一接口: retrieve(query, top_k) -> list[Document]
"""

from typing import Optional

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from .vector_store import get_vector_store


class HybridRetriever:
    """
    BM25 + 向量混合检索 + CrossEncoder 重排。

    检索流程:
        query → [BM25 检索] ──┐
                              ├── RRF 融合 → CrossEncoder 重排 → top_k 文档
        query → [Chroma 向量] ─┘

    用法:
        retriever = HybridRetriever()
        docs = retriever.retrieve("Python 开发经验", top_k=5)
    """

    def __init__(self):
        self._vector_store_mgr = get_vector_store()

        # 检索参数
        self._vector_weight = settings.RAG_HYBRID_VECTOR_WEIGHT
        self._bm25_weight = 1.0 - settings.RAG_HYBRID_VECTOR_WEIGHT
        self._top_k = settings.RAG_TOP_K
        self._rerank_top_n = settings.RAG_TOP_K * 3  # 送入重排的候选数

        # BM25 索引状态
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []
        self._bm25_version: int = -1  # 跟踪向量库变化，按需重建

        # CrossEncoder 重排器（懒加载）
        self._reranker = None

        logger.info(
            f"HybridRetriever 初始化: vector_weight={self._vector_weight}, "
            f"bm25_weight={self._bm25_weight}, top_k={self._top_k}"
        )

    # ============================================================
    # 对外核心接口
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        """
        增强混合检索主入口。

        Args:
            query: 查询文本
            top_k: 返回文档数，默认使用配置值
            use_rerank: 是否启用 CrossEncoder 重排序

        Returns:
            高质量上下文文档列表
        """
        if top_k is None:
            top_k = self._top_k

        if self._vector_store_mgr.count == 0:
            logger.warning("向量库为空，检索返回空列表")
            return []

        # Step 1: 并行检索
        vector_docs = self._vector_search(query, k=self._rerank_top_n)
        bm25_docs = self._bm25_search(query, k=self._rerank_top_n)

        # Step 2: 结果融合
        fused_docs = self._rrf_fusion(
            vector_results=vector_docs,
            bm25_results=bm25_docs,
            k=self._rerank_top_n,
        )

        if not fused_docs:
            logger.warning("混合检索无结果")
            return []

        # Step 3: CrossEncoder 重排序
        if use_rerank:
            reranked = self._rerank(query, fused_docs, top_k)
        else:
            reranked = fused_docs[:top_k]

        logger.info(
            f"检索完成: query='{query[:50]}...' | "
            f"向量={len(vector_docs)} BM25={len(bm25_docs)} "
            f"融合={len(fused_docs)} 输出={len(reranked)}"
        )
        return reranked

    # ============================================================
    # 向量相似度检索
    # ============================================================

    def _vector_search(self, query: str, k: int) -> list[Document]:
        """Chroma 向量相似度检索"""
        try:
            return self._vector_store_mgr.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"向量检索异常: {e}")
            return []

    # ============================================================
    # BM25 关键词检索
    # ============================================================

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        """BM25 关键词检索"""
        # 按需重建 BM25 索引
        self._rebuild_bm25_if_needed()

        if self._bm25_index is None or not self._bm25_docs:
            return []

        try:
            import jieba

            # 中文分词
            tokenized_query = list(jieba.cut(query))
            scores = self._bm25_index.get_scores(tokenized_query)

            # 取 top-k
            if len(scores) == 0:
                return []

            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self._bm25_docs[idx]
                    doc.metadata["bm25_score"] = float(scores[idx])
                    results.append(doc)

            logger.debug(f"BM25 检索: {len(results)} 个结果")
            return results
        except ImportError:
            logger.warning("jieba 未安装，BM25 回退为字符级分词")
            tokenized_query = list(query)
            scores = self._bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self._bm25_docs[idx]
                    doc.metadata["bm25_score"] = float(scores[idx])
                    results.append(doc)
            return results
        except Exception as e:
            logger.error(f"BM25 检索异常: {e}")
            return []

    def _rebuild_bm25_if_needed(self):
        """当向量库文档数变化时，重建 BM25 索引"""
        current_count = self._vector_store_mgr.count
        if self._bm25_version == current_count and self._bm25_index is not None:
            return

        logger.debug(f"重建 BM25 索引: {current_count} 篇文档")
        # 从 Chroma 获取全部文档
        try:
            collection = self._vector_store_mgr.vector_store._collection
            result = collection.get()
            if not result["documents"]:
                self._bm25_index = None
                self._bm25_docs = []
                self._bm25_version = current_count
                return

            self._bm25_docs = [
                Document(
                    page_content=doc,
                    metadata=meta or {},
                )
                for doc, meta in zip(result["documents"], result["metadatas"])
            ]

            # 分词构建 BM25 语料
            try:
                import jieba
                corpus = [list(jieba.cut(doc)) for doc in result["documents"]]
            except ImportError:
                corpus = [list(doc) for doc in result["documents"]]

            self._bm25_index = BM25Okapi(corpus)
            self._bm25_version = current_count
            logger.debug(f"BM25 索引重建完成: {len(corpus)} 篇文档")
        except Exception as e:
            logger.error(f"BM25 索引重建失败: {e}")
            self._bm25_index = None

    # ============================================================
    # RRF 结果融合 (Reciprocal Rank Fusion)
    # ============================================================

    def _rrf_fusion(
        self,
        vector_results: list[Document],
        bm25_results: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion 算法融合两路检索结果。

        原理: RRF_score(doc) = Σ 1/(k + rank_i)
        其中 k 为常数（默认60，可压制极端排名的影响）

        Args:
            vector_results: 向量检索结果列表
            bm25_results: BM25检索结果列表
            k: RRF 常数

        Returns:
            融合并按 RRF 分数降序的文档列表
        """
        # 使用 content hash 作为文档标识
        doc_map: dict[int, Document] = {}  # content_hash -> Document
        rrf_scores: dict[int, float] = {}

        def _content_hash(doc: Document) -> int:
            return hash(doc.page_content[:200])

        # 向量检索排名贡献
        for rank, doc in enumerate(vector_results, start=1):
            h = _content_hash(doc)
            doc_map[h] = doc
            rrf_scores[h] = rrf_scores.get(h, 0.0) + self._vector_weight / (k + rank)

        # BM25 检索排名贡献
        for rank, doc in enumerate(bm25_results, start=1):
            h = _content_hash(doc)
            if h not in doc_map:
                doc_map[h] = doc
            rrf_scores[h] = rrf_scores.get(h, 0.0) + self._bm25_weight / (k + rank)

        # 按 RRF 分数降序排列
        sorted_hashes = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        fused = []
        for h in sorted_hashes:
            doc = doc_map[h]
            doc.metadata["rrf_score"] = rrf_scores[h]
            fused.append(doc)

        logger.debug(f"RRF 融合: 向量={len(vector_results)} + BM25={len(bm25_results)} → {len(fused)}")
        return fused

    # ============================================================
    # CrossEncoder 重排序
    # ============================================================

    def _rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        CrossEncoder 二次重排序，过滤低质量文档。

        使用 BAAI/bge-reranker-v2-m3 模型对候选文档重新打分。

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回文档数

        Returns:
            重排序后的 top_k 文档
        """
        if len(documents) <= top_k:
            return documents

        try:
            # 懒加载 CrossEncoder
            if self._reranker is None:
                from FlagEmbedding import FlagReranker
                model_name = settings.RERANK_MODEL_NAME
                logger.info(f"正在加载 CrossEncoder 重排模型: {model_name}")
                self._reranker = FlagReranker(
                    model_name,
                    use_fp16=False,
                    device=settings.EMBED_DEVICE,
                )
                logger.info("CrossEncoder 加载完成")

            # 构造 query-doc 对
            pairs = [[query, doc.page_content[:1024]] for doc in documents]

            # 打分
            scores = self._reranker.compute_score(pairs, normalize=True)

            # 将分数转为标量列表
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [scores]

            # 附加分数到元数据
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)

            # 按分数降序排序
            sorted_pairs = sorted(
                zip(documents, scores), key=lambda x: x[1], reverse=True
            )

            # 过滤低质量结果（分数 < 阈值 则丢弃）
            threshold = -0.5 if isinstance(scores[0], float) and min(scores) < 0 else None
            if threshold is not None:
                filtered = [
                    doc for doc, score in sorted_pairs
                    if score >= threshold
                ][:top_k]
            else:
                filtered = [doc for doc, _ in sorted_pairs[:top_k]]

            logger.debug(
                f"CrossEncoder 重排: {len(documents)} → {len(filtered)} | "
                f"top_score={sorted_pairs[0][1]:.4f}"
            )
            return filtered

        except ImportError:
            logger.warning("FlagEmbedding 未安装，跳过 CrossEncoder 重排")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"CrossEncoder 重排异常: {e}，回退到 RRF 结果")
            return documents[:top_k]

    # ============================================================
    # 管理接口
    # ============================================================

    def invalidate_bm25_cache(self):
        """强制下次检索时重建 BM25 索引"""
        self._bm25_version = -1
        logger.debug("BM25 缓存已失效")


# ============================================================
# 全局单例
# ============================================================

_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """获取混合检索器全局单例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def retrieve(query: str, top_k: int = 5, use_rerank: bool = True) -> list[Document]:
    """便捷检索函数"""
    return get_retriever().retrieve(query, top_k=top_k, use_rerank=use_rerank)
