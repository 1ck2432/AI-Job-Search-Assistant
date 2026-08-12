"""
config/settings.py - ResumeAgent 全局配置中心
基于 pydantic-settings 从 .env 文件加载所有配置项，提供统一访问入口。
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 基础路径常量
# ============================================================
# 项目根目录 = ResumeAgent/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"


# ============================================================
# LLM Provider 枚举
# ============================================================
LLMProvider = Literal["ollama", "openai", "deepseek"]


# ============================================================
# 全局 Settings 类
# ============================================================
class Settings(BaseSettings):
    """
    全局配置类，自动读取 .env 文件中的环境变量。

    用法:
        from config.settings import settings
        print(settings.LLM_PROVIDER)
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",       # 忽略 .env 中未在此定义的额外字段
    )

    # ==================== LLM Provider 选型 ====================
    LLM_PROVIDER: LLMProvider = Field(
        default="ollama",
        description="LLM 提供商: ollama | openai | deepseek",
    )

    # ==================== Ollama 本地模型 ====================
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址",
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Ollama 对话模型名称",
    )
    OLLAMA_EMBED_MODEL: str = Field(
        default="bge-m3",
        description="Ollama 嵌入模型名称（需先 ollama pull bge-m3）",
    )

    # ==================== OpenAI API ====================
    OPENAI_API_KEY: str = Field(
        default="sk-your-openai-api-key",
        description="OpenAI API Key",
    )
    OPENAI_API_BASE: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API Base URL",
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o",
        description="OpenAI 模型名称",
    )

    # ==================== DeepSeek API ====================
    DEEPSEEK_API_KEY: str = Field(
        default="sk-your-deepseek-api-key",
        description="DeepSeek API Key",
    )
    DEEPSEEK_API_BASE: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API Base URL",
    )
    DEEPSEEK_MODEL: str = Field(
        default="deepseek-chat",
        description="DeepSeek 模型名称",
    )

    # ==================== Embedding 模型 ====================
    EMBED_MODEL_NAME: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 嵌入模型名称（HuggingFace 模型ID）",
    )
    EMBED_DEVICE: str = Field(
        default="cpu",
        description="嵌入模型推理设备: cpu | cuda",
    )

    # ==================== Chroma 向量库 ====================
    CHROMA_PERSIST_DIR: str = Field(
        default="./vector_store/chroma_db",
        description="Chroma 持久化目录",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="resume_knowledge",
        description="Chroma Collection 名称",
    )

    # ==================== RAG 检索参数 ====================
    RAG_TOP_K: int = Field(
        default=5,
        description="检索返回 Top-K 文档数",
    )
    RAG_CHUNK_SIZE: int = Field(
        default=512,
        description="文档切片大小（字符数）",
    )
    RAG_CHUNK_OVERLAP: int = Field(
        default=128,
        description="文档切片重叠长度",
    )
    RAG_HYBRID_VECTOR_WEIGHT: float = Field(
        default=0.6,
        description="混合检索中向量检索的权重 (0~1)，BM25 权重 = 1 - 该值",
    )

    # ==================== CrossEncoder 重排序 ====================
    RERANK_MODEL_NAME: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="CrossEncoder 重排序模型（HuggingFace 模型ID）",
    )

    # ==================== SQLite 数据库 ====================
    SQLITE_DB_PATH: str = Field(
        default="./database/resume_agent.db",
        description="SQLite 数据库文件路径",
    )

    # ==================== 日志 ====================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="日志级别: DEBUG | INFO | WARNING | ERROR",
    )
    LOG_DIR: str = Field(
        default="./logs",
        description="日志文件目录",
    )
    LOG_RETENTION: str = Field(
        default="7 days",
        description="日志保留时长",
    )
    LOG_FORMAT: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Loguru 日志格式",
    )

    # ==================== Gradio 前端 ====================
    GRADIO_SERVER_HOST: str = Field(
        default="127.0.0.1",
        description="Gradio 服务监听地址",
    )
    GRADIO_SERVER_PORT: int = Field(
        default=7860,
        description="Gradio 服务端口",
    )
    GRADIO_TITLE: str = Field(
        default="ResumeAgent - 求职简历智能多Agent协同系统",
        description="Gradio 页面标题",
    )
    GRADIO_THEME: str = Field(
        default="soft",
        description="Gradio 主题",
    )

    # ==================== 文件上传 ====================
    UPLOAD_DIR: str = Field(
        default=str(UPLOAD_DIR),
        description="文件上传目录",
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=10,
        description="最大上传文件大小（MB）",
    )

    # ==================== 辅助属性 ====================

    @property
    def is_ollama(self) -> bool:
        """是否使用 Ollama 本地模型"""
        return self.LLM_PROVIDER == "ollama"

    @property
    def is_openai(self) -> bool:
        """是否使用 OpenAI API"""
        return self.LLM_PROVIDER == "openai"

    @property
    def is_deepseek(self) -> bool:
        """是否使用 DeepSeek API"""
        return self.LLM_PROVIDER == "deepseek"

    @property
    def chroma_persist_path(self) -> str:
        """返回 Chroma 持久化的绝对路径"""
        path = Path(self.CHROMA_PERSIST_DIR)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    @property
    def sqlite_db_abs_path(self) -> str:
        """返回 SQLite 数据库文件的绝对路径"""
        path = Path(self.SQLITE_DB_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    @property
    def log_dir_abs_path(self) -> str:
        """返回日志目录的绝对路径"""
        path = Path(self.LOG_DIR)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    @property
    def upload_dir_abs_path(self) -> str:
        """返回上传目录的绝对路径"""
        path = Path(self.UPLOAD_DIR)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())


# ============================================================
# 全局单例
# ============================================================
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取 Settings 全局单例（惰性初始化）。
    首次调用时加载 .env 配置，后续调用返回同一实例。
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# 模块级便捷引用（推荐在代码中统一使用 from config.settings import settings）
settings: Settings = get_settings()
