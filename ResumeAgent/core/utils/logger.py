"""
core/utils/logger.py - Loguru 日志辅助工具
提供进度装饰器、耗时统计、异常捕获、健康检查等便捷日志函数。
"""

import time
import functools
import traceback
from typing import Callable, Any, Optional, TypeVar
from pathlib import Path

from loguru import logger

# ============================================================
# 装饰器
# ============================================================

F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(func: F) -> F:
    """
    装饰器：自动记录被装饰函数的执行耗时。

    用法:
        @log_execution_time
        def my_slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"[{func.__name__}] 执行完成 | 耗时: {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"[{func.__name__}] 执行异常 | 耗时: {elapsed:.3f}s | 错误: {e}")
            raise

    return wrapper  # type: ignore[return-value]


def safe_execute(
    default_return: Any = None,
    error_msg: str = "操作失败，请查看日志",
    reraise: bool = False,
) -> Callable[[F], F]:
    """
    装饰器：安全执行函数，捕获所有异常并记录日志。
    适用于 Gradio 回调等不希望因异常而崩溃的场景。

    Args:
        default_return: 异常时返回的默认值
        error_msg:     用户可见的错误消息前缀
        reraise:       是否重新抛出异常

    用法:
        @safe_execute(default_return="❌ 处理失败", error_msg="评分分析出错")
        def my_callback(x):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 异常简要信息输出到控制台，完整堆栈仅写入错误日志文件
                logger.error(
                    f"[{func.__name__}] {error_msg}: {e}"
                )
                # 完整堆栈追溯写入文件日志（不输出到控制台）
                logger.opt(depth=1, exception=True).debug(
                    f"[{func.__name__}] 异常详情"
                )
                if reraise:
                    raise
                return default_return
        return wrapper  # type: ignore[return-value]
    return decorator


def log_step(step_name: str) -> None:
    """
    在关键流程步骤处埋入日志，便于追踪业务流转。

    用法:
        log_step("开始解析简历")
    """
    logger.info(f"[Step] {step_name}")


# ============================================================
# 上下文管理器
# ============================================================

class TimingContext:
    """
    耗时统计上下文管理器。

    用法:
        with TimingContext("LLM 推理"):
            result = llm.chat(...)
    """

    def __init__(self, label: str, level: str = "INFO"):
        self.label = label
        self.level = level.upper()
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        logger.debug(f"[Timing] {self.label} 开始...")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        log_func = getattr(logger, self.level.lower(), logger.info)
        log_func(f"[Timing] {self.label} 完成 | 耗时: {elapsed:.3f}s")


# ============================================================
# 健康检查
# ============================================================

class HealthStatus:
    """健康检查结果模型。"""
    def __init__(self):
        self.checks: dict[str, dict] = {}

    def add(self, name: str, passed: bool, detail: str = "", warn: bool = False):
        self.checks[name] = {
            "passed": passed,
            "detail": detail,
            "warning": warn,
        }

    @property
    def all_ok(self) -> bool:
        return all(
            c["passed"] or c["warning"]
            for c in self.checks.values()
        )

    @property
    def critical_ok(self) -> bool:
        critical = {k: v for k, v in self.checks.items() if not v.get("warning")}
        return all(c["passed"] for c in critical.values())

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "  系统健康检查", "=" * 60]
        for name, check in self.checks.items():
            icon = "✅" if check["passed"] else ("⚠️" if check.get("warning") else "❌")
            status = f"{icon} {name}"
            if check["detail"]:
                status += f": {check['detail']}"
            lines.append(status)
        lines.append("=" * 60)
        return "\n".join(lines)


def run_health_check(llm_provider: Optional[str] = None) -> HealthStatus:
    """
    执行系统启动前健康检查。

    检查项：
    1. Python 关键依赖是否可用
    2. 文件系统权限（日志/上传/导出/向量库目录）
    3. SQLite 数据库连接与表结构
    4. ChromaDB 向量库连接
    5. LLM 连接可用性（可选）

    Args:
        llm_provider: LLM 提供商标识，用于连通性测试

    Returns:
        HealthStatus 对象
    """
    status = HealthStatus()
    logger.info("正在执行系统健康检查...")

    # --- 1. 关键依赖 ---
    try:
        import gradio as gr  # noqa: F401
        status.add("Gradio", True, f"v{gr.__version__}")
    except ImportError as e:
        status.add("Gradio", False, str(e))

    try:
        import langchain_core  # noqa: F401
        status.add("LangChain", True, f"已安装")
    except ImportError as e:
        status.add("LangChain", False, str(e))

    try:
        import chromadb  # noqa: F401
        status.add("ChromaDB", True, f"v{chromadb.__version__}")
    except ImportError as e:
        status.add("ChromaDB", False, str(e))

    try:
        import plotly  # noqa: F401
        status.add("Plotly", True, f"v{plotly.__version__}")
    except ImportError as e:
        status.add("Plotly", False, str(e))

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        status.add("Sentence-Transformers", True)
    except ImportError as e:
        status.add("Sentence-Transformers", True, f"未安装({e})", warn=True)

    # --- 2. 文件系统目录 ---
    from config.settings import settings
    dir_map = {
        "日志目录": Path(settings.log_dir_abs_path),
        "上传目录": Path(settings.upload_dir_abs_path),
        "导出目录": Path("exports"),
        "向量库目录": Path(settings.chroma_persist_path),
    }
    for name, d in dir_map.items():
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            status.add(name, True, str(d))
        except Exception as e:
            status.add(name, False, str(e))

    # --- 3. SQLite 数据库 ---
    try:
        from database.db import get_db
        db = get_db()
        tables = db._get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        expected = {"resumes", "jds", "optimizations", "interview_records"}
        missing = expected - set(table_names)
        if missing:
            status.add("SQLite数据库", True, f"缺少表: {missing}", warn=True)
        else:
            status.add("SQLite数据库", True, f"{len(table_names)} 张表就绪")
    except Exception as e:
        status.add("SQLite数据库", False, str(e))

    # --- 4. ChromaDB 向量库 ---
    try:
        from core.rag.vector_store import get_vector_store
        vs = get_vector_store()
        count = vs.count
        status.add("ChromaDB向量库", True, f"{count} 条记录")
    except Exception as e:
        status.add("ChromaDB向量库", True, str(e), warn=True)

    # --- 5. LLM 连通性测试（可选） ---
    if llm_provider:
        try:
            from core.llm import get_llm
            from core.llm.base import ChatMessage
            llm = get_llm()
            resp = llm.chat([ChatMessage.user("ping")], max_tokens=5)
            status.add(f"LLM({llm_provider})", True, "连通正常")
        except Exception as e:
            status.add(f"LLM({llm_provider})", False, str(e))

    # 输出汇总
    for line in status.summary().split("\n"):
        logger.info(line)

    return status


# ============================================================
# 应用生命周期
# ============================================================

_startup_time: float = 0.0


def mark_startup():
    """标记应用启动时刻。"""
    global _startup_time
    _startup_time = time.time()


def get_uptime() -> str:
    """获取应用运行时长。"""
    if _startup_time == 0:
        return "未知"
    delta = time.time() - _startup_time
    if delta < 60:
        return f"{delta:.0f} 秒"
    elif delta < 3600:
        return f"{delta / 60:.1f} 分钟"
    else:
        return f"{delta / 3600:.1f} 小时"
