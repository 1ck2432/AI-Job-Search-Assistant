"""
core/rag/vector_store.py - Chroma 向量库管理

功能:
- Chroma 向量库实例化（持久化存储）
- 文档批量入库（Embedding + 存入 Chroma）
- 按文档ID/元数据条件删除
- 清空向量库
- Collection 基础信息查询
"""

import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from loguru import logger

from config.settings import settings
from .embedding import get_embedding_model


class VectorStoreManager:
    """
    Chroma 向量库管理器（单例）。

    用法:
        vs = VectorStoreManager()
        vs.add_documents(docs)
        results = vs.similarity_search("Python 开发经验", k=5)
        vs.delete_by_source("resume.pdf")
    """

    _instance: Optional["VectorStoreManager"] = None

    def __new__(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 配置
        self._persist_dir = settings.chroma_persist_path
        self._collection_name = settings.CHROMA_COLLECTION_NAME

        # Chroma 持久化客户端
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Embedding 模型（单例）
        self._embedding = get_embedding_model()

        # LangChain Chroma 向量库
        self._vector_store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embedding,
        )

        logger.info(
            f"Chroma 向量库初始化: collection={self._collection_name}, "
            f"path={self._persist_dir}, "
            f"docs={self._vector_store._collection.count()}"
        )

    # ----------------------------------------------------------
    # 文档入库
    # ----------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 50,
    ) -> list[str]:
        """
        批量将 LangChain Document 写入向量库。

        Args:
            documents: Document 列表
            batch_size: 每批入库数量

        Returns:
            入库的文档ID列表
        """
        if not documents:
            logger.warning("add_documents: 文档列表为空，跳过")
            return []

        # 为每个文档生成唯一 ID（若未提供）
        ids = []
        for doc in documents:
            doc_id = doc.metadata.get("doc_id", str(uuid.uuid4().hex[:12]))
            doc.metadata["doc_id"] = doc_id
            ids.append(doc_id)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            try:
                self._vector_store.add_documents(batch_docs, ids=batch_ids)
                logger.debug(f"向量入库进度: {min(i+batch_size, total)}/{total}")
            except Exception as e:
                logger.error(f"向量入库失败 [批次 {i//batch_size}]: {e}")

        collection_count = self._vector_store._collection.count()
        logger.info(f"向量入库完成: {total} 个文档 | 当前总量={collection_count}")
        return ids

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        直接以文本列表方式入库（自动生成 Document）。

        Args:
            texts: 文本列表
            metadatas: 元数据列表（与 texts 等长）
            ids: 文档ID列表（缺省自动生成）

        Returns:
            文档ID列表
        """
        if ids is None:
            ids = [str(uuid.uuid4().hex[:12]) for _ in texts]

        if metadatas is None:
            metadatas = [{}] * len(texts)

        for meta, doc_id in zip(metadatas, ids):
            meta["doc_id"] = doc_id

        try:
            self._vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
            logger.info(f"文本入库完成: {len(texts)} 条")
            return ids
        except Exception as e:
            logger.error(f"文本入库失败: {e}")
            raise

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """
        向量相似度检索。

        Args:
            query: 查询文本
            k: 返回文档数

        Returns:
            相似文档列表（附带相似度分数在 metadata['score'] 中）
        """
        try:
            return self._vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        向量相似度检索（带分数）。

        Args:
            query: 查询文本
            k: 返回文档数

        Returns:
            [(Document, 相似度分数)] 列表
        """
        try:
            return self._vector_store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            logger.error(f"向量检索(带分数)失败: {e}")
            return []

    # ----------------------------------------------------------
    # 删除操作
    # ----------------------------------------------------------

    def delete_by_ids(self, ids: list[str]) -> int:
        """
        按文档 ID 列表删除。

        Args:
            ids: 文档 ID 列表

        Returns:
            实际删除的数量
        """
        if not ids:
            return 0

        before = self._vector_store._collection.count()
        try:
            self._vector_store._collection.delete(ids=ids)
            after = self._vector_store._collection.count()
            deleted = before - after
            logger.info(f"向量删除完成: {deleted} 个文档 (by ids)")
            return deleted
        except Exception as e:
            logger.error(f"向量删除失败: {e}")
            return 0

    def delete_by_source(self, source: str) -> int:
        """
        按来源文件删除该文件的所有切片。

        Args:
            source: 来源文件路径/标识

        Returns:
            删除数量
        """
        before = self._vector_store._collection.count()
        try:
            self._vector_store._collection.delete(
                where={"source": source}
            )
            after = self._vector_store._collection.count()
            deleted = before - after
            logger.info(f"向量删除完成: {deleted} 个文档 (source={source})")
            return deleted
        except Exception as e:
            logger.error(f"按来源删除失败: {e}")
            return 0

    def delete_by_filter(self, filter_dict: dict) -> int:
        """
        按元数据条件删除。

        Args:
            filter_dict: Chroma where 条件

        Returns:
            删除数量
        """
        before = self._vector_store._collection.count()
        try:
            self._vector_store._collection.delete(where=filter_dict)
            after = self._vector_store._collection.count()
            deleted = before - after
            logger.info(f"向量删除完成: {deleted} 个文档 (filter={filter_dict})")
            return deleted
        except Exception as e:
            logger.error(f"按条件删除失败: {e}")
            return 0

    # ----------------------------------------------------------
    # 管理操作
    # ----------------------------------------------------------

    def clear(self) -> bool:
        """
        清空整个 Collection（不可逆！）。

        Returns:
            是否成功
        """
        before = self._vector_store._collection.count()
        try:
            # 删除 collection 后重建
            self._client.delete_collection(self._collection_name)
            self._vector_store = Chroma(
                client=self._client,
                collection_name=self._collection_name,
                embedding_function=self._embedding,
            )
            logger.warning(f"向量库已清空，原 {before} 个文档已删除")
            return True
        except Exception as e:
            logger.error(f"向量库清空失败: {e}")
            return False

    @property
    def count(self) -> int:
        """当前 Collection 中文档数量"""
        try:
            return self._vector_store._collection.count()
        except Exception:
            return 0

    def get_collection_info(self) -> dict:
        """获取 Collection 概览信息"""
        return {
            "name": self._collection_name,
            "document_count": self.count,
            "persist_dir": self._persist_dir,
        }

    @property
    def vector_store(self) -> Chroma:
        """获取底层 Chroma 实例（供 retriever 高级检索用）"""
        return self._vector_store


# ============================================================
# 便捷函数
# ============================================================

def get_vector_store() -> VectorStoreManager:
    """获取向量库管理器单例"""
    return VectorStoreManager()
