"""
core/rag/text_splitter.py - 文档智能切片

功能:
- 递归字符分割 (RecursiveCharacterTextSplitter)
- 标题感知分层切片：对 Markdown/简历格式按标题层级切分
- 支持自定义 chunk_size、chunk_overlap
- 输出带元数据（标题、来源文件）的 LangChain Document 列表
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from loguru import logger

from config.settings import settings


class DocumentSplitter:
    """
    文档智能切片器。

    策略:
    1. 先尝试按标题分层切片（适用于有结构的内容）
    2. 对每层内容再进行递归字符分割，保证 chunk 大小合理

    用法:
        splitter = DocumentSplitter(chunk_size=512, chunk_overlap=128)
        docs = splitter.split("这是一篇长文档...", metadata={"source": "resume.pdf"})
    """

    # Markdown 标题层级配置（也兼容常见简历格式）
    HEADERS_TO_SPLIT_ON = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Args:
            chunk_size: 每个切片的字符数上限，默认读取配置
            chunk_overlap: 切片之间重叠字符数，默认读取配置
        """
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        # 递归字符分割器
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " ", ""],
            length_function=len,
            keep_separator=True,
        )

        logger.debug(
            f"DocumentSplitter 初始化: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    # ----------------------------------------------------------
    # 核心切片方法
    # ----------------------------------------------------------

    def split(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        将文本切分为多个 LangChain Document 片段。

        流程：
        1. 尝试标题分层切片
        2. 若标题分层后仍有过长段落，递归字符二次切分

        Args:
            text: 原始文本
            metadata: 元数据（来源文件、类别等）

        Returns:
            List[Document]: 切片后的文档列表
        """
        base_meta = metadata or {}
        base_meta.setdefault("chunk_size", self.chunk_size)

        # 先尝试标题分层切片
        md_docs = self._split_by_headers(text, base_meta)

        # 对标题切片结果递归字符二次切分
        all_docs: list[Document] = []
        for doc in md_docs:
            if len(doc.page_content) > self.chunk_size * 1.2:
                # 过长段落，二次递归切分
                sub_docs = self._char_splitter.split_documents([doc])
                all_docs.extend(sub_docs)
            else:
                all_docs.append(doc)

        # 为每个切片添加序号
        for i, doc in enumerate(all_docs):
            doc.metadata["chunk_index"] = i

        logger.info(
            f"文档切片完成: {len(all_docs)} 个片段 | "
            f"平均长度={sum(len(d.page_content) for d in all_docs)//max(len(all_docs),1)} 字符"
        )
        return all_docs

    # ----------------------------------------------------------
    # 标题感知分层
    # ----------------------------------------------------------

    def _split_by_headers(self, text: str, base_meta: dict) -> list[Document]:
        """
        按标题层级切分文档。

        先尝试 MarkdownHeaderTextSplitter，
        若无法识别标题结构，则回退为简单按段落切分。

        Args:
            text: 输入文本
            base_meta: 基础元数据

        Returns:
            按标题分层的 Document 列表
        """
        try:
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
                strip_headers=False,
            )
            docs = md_splitter.split_text(text)

            if len(docs) > 1:
                logger.debug(f"标题分层切分成功: {len(docs)} 个段落")
                for doc in docs:
                    doc.metadata.update(base_meta)
                return docs
        except Exception as e:
            logger.debug(f"标题分层切分未生效: {e}")

        # 回退：整篇作为一个 Document
        return [Document(page_content=text, metadata=base_meta)]

    def split_batch(
        self,
        texts: dict[str, str],
    ) -> list[Document]:
        """
        批量切片：对多个文本分别切片并汇总。

        Args:
            texts: {来源标识: 文本内容} 字典

        Returns:
            所有切片汇总列表
        """
        all_docs: list[Document] = []
        for source, text in texts.items():
            docs = self.split(text, metadata={"source": source})
            all_docs.extend(docs)
        logger.info(f"批量切片完成: 总计 {len(all_docs)} 个片段")
        return all_docs


# ============================================================
# 便捷函数
# ============================================================

def split_document(
    text: str,
    metadata: Optional[dict] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """便捷函数：切片单个文档"""
    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split(text, metadata)
