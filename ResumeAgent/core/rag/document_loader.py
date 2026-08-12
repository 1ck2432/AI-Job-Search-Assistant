"""
core/rag/document_loader.py - 文档加载与文本清洗

功能:
- 统一加载 PDF/Word/TXT 文档
- 文本清洗：去除多余换行、乱码字符、水印冗余文本
- 输出清洗后的纯文本供后续切片使用
"""

import re
from pathlib import Path
from typing import Optional

from loguru import logger

from core.utils.file_parser import parse_file, SUPPORTED_EXTENSIONS


class DocumentLoader:
    """
    文档加载器：解析文件 + 文本清洗。

    用法:
        loader = DocumentLoader()
        text = loader.load("resume.pdf")
        texts = loader.load_batch(["resume.pdf", "jd.docx"])
    """

    # 常见水印/页脚模式（正则）
    WATERMARK_PATTERNS = [
        r"Powered by .*",
        r"Created with .*",
        r"Copyright ©.*",
        r"Confidential",
        r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页",
        r"Page\s+\d+\s+of\s+\d+",
        r"^\d+/\d+$",
    ]

    def __init__(self):
        self._compiled_watermarks = [re.compile(p, re.IGNORECASE) for p in self.WATERMARK_PATTERNS]

    # ----------------------------------------------------------
    # 单文件加载
    # ----------------------------------------------------------

    def load(self, file_path: str) -> str:
        """
        加载单个文件并清洗文本。

        Args:
            file_path: 文件路径

        Returns:
            清洗后的纯文本
        """
        logger.debug(f"开始加载文档: {file_path}")
        raw_text = parse_file(file_path)
        cleaned = self._clean_text(raw_text)
        logger.info(f"文档加载完成: {Path(file_path).name} | 原始={len(raw_text)}字符 | 清洗后={len(cleaned)}字符")
        return cleaned

    # ----------------------------------------------------------
    # 批量加载
    # ----------------------------------------------------------

    def load_batch(self, file_paths: list[str]) -> dict[str, str]:
        """
        批量加载多个文件，返回 {文件路径: 清洗文本}。

        Args:
            file_paths: 文件路径列表

        Returns:
            字典：文件路径 -> 清洗文本
        """
        results: dict[str, str] = {}
        for fp in file_paths:
            try:
                results[fp] = self.load(fp)
            except Exception as e:
                logger.error(f"加载失败 [{fp}]: {e}")
        logger.info(f"批量加载完成: {len(results)}/{len(file_paths)}")
        return results

    # ----------------------------------------------------------
    # 文本清洗
    # ----------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        公开接口：对原始文本执行多步清洗流水线。
        供 agent_nodes 等外部模块直接调用。
        """
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """
        对原始文本执行多步清洗流水线。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 1. 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. 去除乱码控制字符（保留常用空白和可打印字符）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 3. 去除水印/页脚行
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # 跳过空行（后面会统一处理）
            if not stripped:
                cleaned_lines.append("")
                continue
            # 跳过水印行
            if self._is_watermark(stripped):
                continue
            # 跳过纯数字行（可能是页码）
            if stripped.isdigit():
                continue
            cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)

        # 4. 合并连续空行（最多保留 1 个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. 去除首尾空白
        text = text.strip()

        return text

    def _is_watermark(self, line: str) -> bool:
        """判断某行是否为水印/页脚文本"""
        for pattern in self._compiled_watermarks:
            if pattern.search(line):
                return True
        return False


# ============================================================
# 便捷函数
# ============================================================

def load_document(file_path: str) -> str:
    """便捷函数：加载并清洗单个文档"""
    loader = DocumentLoader()
    return loader.load(file_path)


def load_documents(file_paths: list[str]) -> dict[str, str]:
    """便捷函数：批量加载文档"""
    loader = DocumentLoader()
    return loader.load_batch(file_paths)
