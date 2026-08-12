"""
core/rag/embedding.py - BGE-M3 嵌入模型管理

支持加载方式:
- HuggingFace 本地加载 (sentence-transformers，默认)
- 兼容 Ollama Embedding API（若用户通过 Ollama 部署了 bge-m3）

单例模式，全局复用，避免重复加载大模型。
"""

from typing import Optional

from loguru import logger
from langchain_core.embeddings import Embeddings

from config.settings import settings


# ============================================================
# HuggingFace 本地 BGE-M3
# ============================================================

class BGEM3Embeddings(Embeddings):
    """
    基于 sentence-transformers 的 BGE-M3 LangChain Embeddings 封装。

    特性：
    - 懒加载：首次调用 embed_documents/embed_query 时才加载模型
    - 单例模型：全局只加载一次
    """

    _model = None
    _device: Optional[str] = None

    def __init__(self):
        self._model_name = settings.EMBED_MODEL_NAME
        self._device = settings.EMBED_DEVICE

    def _lazy_init(self):
        """懒加载：首次使用时加载模型"""
        cls = type(self)
        if cls._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"正在加载 BGE-M3 模型: {self._model_name} (device={self._device})")
            cls._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            cls._device = self._device
            dim = cls._model.get_sentence_embedding_dimension()
            logger.info(f"BGE-M3 加载完成，向量维度={dim}")
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请执行: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量将文档文本转为嵌入向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量为 float 列表
        """
        self._lazy_init()
        embeddings = type(self)._model.encode(
            texts,
            normalize_embeddings=True,    # BGE 推荐归一化
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        将单条查询文本转为嵌入向量。

        Args:
            text: 查询文本

        Returns:
            向量 (float 列表)
        """
        self._lazy_init()
        embedding = type(self)._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# ============================================================
# Ollama 远程 Embedding API
# ============================================================

class OllamaEmbeddings(Embeddings):
    """
    通过 Ollama API 获取 Embedding。
    需要先在 Ollama 中拉取 bge-m3 模型: ollama pull bge-m3
    """

    def __init__(self):
        import httpx
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.OLLAMA_EMBED_MODEL
        self._client = httpx.Client(timeout=10.0)

        # 连接探测：发送一次轻量 embedding 请求验证可用性
        try:
            self._embed("test")
            logger.info(f"Ollama Embedding 连接成功: {self._model} @ {self._base_url}")
        except Exception as e:
            self._client.close()
            raise ConnectionError(f"Ollama Embedding 不可用: {e}")

    def _embed(self, text: str) -> list[float]:
        """调用 Ollama Embedding API"""
        try:
            resp = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            logger.error(f"Ollama Embedding 请求失败: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入"""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条查询嵌入"""
        return self._embed(text)


# ============================================================
# 工厂函数
# ============================================================

_embedding_instance: Optional[Embeddings] = None


def get_embedding_model() -> Embeddings:
    """
    获取全局单例 Embedding 模型。

    策略：
    - ollama 模式下：优先尝试 OllamaEmbeddings，失败回退本地 BGE-M3
    - openai/deepseek 模式下：直接使用本地 BGE-M3（云端场景无需依赖 Ollama）

    Returns:
        LangChain Embeddings 实例
    """
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    if settings.is_ollama:
        try:
            _embedding_instance = OllamaEmbeddings()
            logger.info(f"使用 Ollama Embedding: {settings.OLLAMA_EMBED_MODEL}")
            return _embedding_instance
        except Exception as e:
            logger.warning(f"Ollama Embedding 不可用 ({e})，回退到本地 BGE-M3")

    _embedding_instance = BGEM3Embeddings()
    logger.info(f"使用本地 Embedding: {settings.EMBED_MODEL_NAME}")
    return _embedding_instance
