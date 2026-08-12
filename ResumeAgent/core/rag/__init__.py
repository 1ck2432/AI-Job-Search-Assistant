"""
core/rag/__init__.py - 增强 RAG 全链路模块统一入口

提供完整的检索增强生成流水线:
  文档加载 → 文本切片 → 向量化入库 → 混合检索 → 重排序

用法:
    from core.rag import (
        DocumentLoader, DocumentSplitter,
        get_vector_store, get_retriever,
        load_and_index_document, search_knowledge,
    )
"""

from .document_loader import DocumentLoader, load_document, load_documents
from .text_splitter import DocumentSplitter, split_document
from .embedding import get_embedding_model
from .vector_store import VectorStoreManager, get_vector_store
from .retriever import HybridRetriever, get_retriever, retrieve

from loguru import logger


# ============================================================
# 一站式入库函数
# ============================================================

def load_and_index_document(
    file_path: str,
    metadata: dict | None = None,
) -> list[str]:
    """
    一站式：加载文档 → 切片 → 向量化 → 入库。

    Args:
        file_path: 文件路径
        metadata: 额外元数据

    Returns:
        入库文档 ID 列表
    """
    logger.info(f"一站式入库开始: {file_path}")

    # 1. 加载清洗
    loader = DocumentLoader()
    text = loader.load(file_path)

    # 2. 智能切片
    base_meta = {"source": file_path}
    if metadata:
        base_meta.update(metadata)

    splitter = DocumentSplitter()
    docs = splitter.split(text, metadata=base_meta)

    # 3. 向量化入库
    vs = get_vector_store()
    ids = vs.add_documents(docs)

    # 4. 通知检索器缓存失效
    retriever = get_retriever()
    retriever.invalidate_bm25_cache()

    logger.info(f"一站式入库完成: {file_path} → {len(ids)} 个文档")
    return ids


def search_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """
    便捷知识检索，返回带分数的上下文文档。

    Args:
        query: 查询文本
        top_k: 返回数量

    Returns:
        [{"content": str, "source": str, "score": float}, ...]
    """
    retriever = get_retriever()
    docs = retriever.retrieve(query, top_k=top_k)

    results = []
    for doc in docs:
        score = doc.metadata.get("rerank_score") or doc.metadata.get("rrf_score", 0)
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "metadata": doc.metadata,
        })
    return results


# 模块导出清单
__all__ = [
    # 文档处理
    "DocumentLoader",
    "load_document",
    "load_documents",
    "DocumentSplitter",
    "split_document",
    # 嵌入模型
    "get_embedding_model",
    # 向量库
    "VectorStoreManager",
    "get_vector_store",
    # 增强检索
    "HybridRetriever",
    "get_retriever",
    "retrieve",
    # 一站式
    "load_and_index_document",
    "search_knowledge",
]
