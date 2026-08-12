"""
main.py - ResumeAgent 系统入口
负责日志初始化、健康检查、目录检查、启动 Gradio Web 服务。
"""

# ============================================================
# 国内 HuggingFace 镜像（必须在导入 HF 依赖前设置）
# ============================================================
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import atexit
import signal
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保子模块可以独立导入
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from config.settings import settings
from core.utils.logger import run_health_check, mark_startup, get_uptime


# ============================================================
# 日志初始化
# ============================================================

def init_logging() -> None:
    """
    初始化 Loguru 日志系统。
    - 控制台输出：彩色格式
    - 文件输出：按天轮转，自动保留指定天数
    """
    logger.remove()  # 移除默认 handler

    # 控制台日志（彩色）
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # 文件日志（按天轮转）
    log_dir = Path(settings.log_dir_abs_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "resume_agent_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        rotation="00:00",          # 每天午夜轮转
        retention=settings.LOG_RETENTION,
        encoding="utf-8",
        enqueue=True,              # 异步写入，避免阻塞主线程
    )

    # 异常专用日志
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format=settings.LOG_FORMAT,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )

    logger.info(f"日志系统初始化完成 | 级别={settings.LOG_LEVEL} | 目录={log_dir}")


# ============================================================
# 目录检查
# ============================================================

def ensure_directories() -> None:
    """
    确保运行时所需的所有目录存在。
    """
    dirs_to_check = [
        settings.upload_dir_abs_path,
        settings.chroma_persist_path,
        str(Path(settings.sqlite_db_abs_path).parent),
        settings.log_dir_abs_path,
        str(PROJECT_ROOT / "exports"),
    ]
    for d in dirs_to_check:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("运行时目录检查完成")


# ============================================================
# 优雅关闭
# ============================================================

_shutdown_registered = False


def _graceful_shutdown(signum=None, frame=None):
    """优雅关闭处理：记录日志、清理资源。"""
    signal_name = signal.Signals(signum).name if signum else "MANUAL"
    logger.warning(f"收到关闭信号 [{signal_name}]，正在优雅退出...")

    # 1. 清理临时文件
    try:
        import tempfile
        import shutil
        tmp_dir = Path(tempfile.gettempdir()) / "resume_agent"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("临时文件已清理")
    except Exception:
        pass

    # 2. 记录运行时长
    logger.info(f"系统已运行 {get_uptime()}，正在关闭...")

    logger.info("ResumeAgent 已安全退出")
    sys.exit(0)


def register_shutdown_handlers():
    """注册进程信号处理和 atexit 回调。"""
    global _shutdown_registered
    if _shutdown_registered:
        return
    _shutdown_registered = True

    # SIGINT (Ctrl+C) / SIGTERM
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    # 正常退出时的清理
    atexit.register(lambda: logger.info("ResumeAgent 进程退出"))

    logger.debug("关闭信号处理器已注册")


# ============================================================
# 启动 Web 服务
# ============================================================

def launch_webui() -> None:
    """
    启动 Gradio Web UI。
    实际界面逻辑在 webui/gradio_ui.py 中实现，此处仅做入口调用。
    """
    try:
        import gradio as gr
        from webui.gradio_ui import create_ui

        css = """
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
        """
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald")

        demo = create_ui()
        logger.info(f"Gradio 界面构建完成，正在启动服务...")
        demo.launch(
            server_name=settings.GRADIO_SERVER_HOST,
            server_port=settings.GRADIO_SERVER_PORT,
            share=False,
            show_error=True,
            theme=theme,
            css=css,
        )
    except ImportError as e:
        logger.error(f"无法导入 Gradio UI 模块: {e}")
        logger.warning("请确认 webui/gradio_ui.py 已创建且所有依赖已安装")
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Gradio 启动失败: {e}")
        raise


# ============================================================
# 打印系统信息
# ============================================================

def print_banner() -> None:
    """打印系统启动横幅。"""
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║  🎯 ResumeAgent - AI 多智能体求职助手                   ║
║  基于 LangGraph 多智能体协作的智能求职辅助系统           ║
╠══════════════════════════════════════════════════════════╣
║  LLM: {settings.LLM_PROVIDER:<20s}                         ║
║  Gradio: http://{settings.GRADIO_SERVER_HOST}:{settings.GRADIO_SERVER_PORT:<5d}                     ║
╚══════════════════════════════════════════════════════════╝
"""
    logger.info(banner)


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """
    ResumeAgent 系统主入口。
    执行顺序：初始化日志 → 检查目录 → 健康检查 → 注册关闭 → 启动 Web
    """
    # 1. 初始化日志
    init_logging()

    # 2. 打印横幅
    print_banner()

    # 3. 注册关闭处理
    register_shutdown_handlers()

    # 4. 确保运行时目录存在
    ensure_directories()

    # 5. 输出当前关键配置
    logger.info(f"LLM 模式: {settings.LLM_PROVIDER}")
    if settings.is_ollama:
        logger.info(f"  → Ollama: {settings.OLLAMA_MODEL} @ {settings.OLLAMA_BASE_URL}")
    elif settings.is_openai:
        logger.info(f"  → OpenAI: {settings.OPENAI_MODEL}")
    elif settings.is_deepseek:
        logger.info(f"  → DeepSeek: {settings.DEEPSEEK_MODEL}")
    logger.info(f"向量库: ChromaDB @ {settings.chroma_persist_path}")
    logger.info(f"数据库: SQLite @ {settings.sqlite_db_abs_path}")
    logger.info(f"Embedding: {settings.EMBED_MODEL_NAME} @ {settings.EMBED_DEVICE}")

    # 6. 健康检查（非阻塞，仅报告）
    try:
        health = run_health_check(llm_provider=settings.LLM_PROVIDER)
        if not health.all_ok:
            logger.warning("部分健康检查未通过，系统可能功能受限")
    except Exception as e:
        logger.warning(f"健康检查执行异常: {e}，跳过检查继续启动")

    # 7. 标记启动时间
    mark_startup()

    # 8. 启动 Gradio Web 服务
    launch_webui()


if __name__ == "__main__":
    main()
