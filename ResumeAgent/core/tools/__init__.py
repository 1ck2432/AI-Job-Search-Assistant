"""
core.tools - 工具模块

提供分数计算、文件导出、数据库管理等通用工具。

导出:
    - score_tool:   分数计算格式化、雷达图数据构造
    - file_export:  优化简历导出 docx、面试报告生成 txt
    - sqlite_db:    历史记录增删查改统一管理
"""

# ============================================================
# 分数 & 雷达图
# ============================================================
from core.tools.score_tool import (
    # 分数计算
    calculate_weighted_score,
    score_to_grade,
    compare_scores,
    # 格式化
    format_match_report,
    format_compare_report,
    format_score_bar,
    scores_to_dict,
    # 雷达图
    build_radar_chart_data,
    build_radar_comparison_data,
    build_plotly_figure_config,
    # 常量
    DEFAULT_WEIGHTS,
    GRADE_THRESHOLDS,
    GRADE_COLORS,
    RADAR_LABELS,
)

# ============================================================
# 文件导出
# ============================================================
from core.tools.file_export import (
    export_resume_to_docx,
    export_interview_report_to_txt,
    export_match_report_to_txt,
    export_compare_report_to_txt,
    get_export_dir,
    set_export_dir,
)

# ============================================================
# 数据库管理
# ============================================================
from core.tools.sqlite_db import (
    RecordManager,
    ResumeRepo,
    JDRepo,
    OptimizationRepo,
    InterviewRepo,
    PageResult,
    get_record_manager,
)

# ============================================================
# 模块级别 __all__
# ============================================================
__all__ = [
    # score_tool
    "calculate_weighted_score",
    "score_to_grade",
    "compare_scores",
    "format_match_report",
    "format_compare_report",
    "format_score_bar",
    "scores_to_dict",
    "build_radar_chart_data",
    "build_radar_comparison_data",
    "build_plotly_figure_config",
    "DEFAULT_WEIGHTS",
    "GRADE_THRESHOLDS",
    "GRADE_COLORS",
    "RADAR_LABELS",
    # file_export
    "export_resume_to_docx",
    "export_interview_report_to_txt",
    "export_match_report_to_txt",
    "export_compare_report_to_txt",
    "get_export_dir",
    "set_export_dir",
    # sqlite_db
    "RecordManager",
    "ResumeRepo",
    "JDRepo",
    "OptimizationRepo",
    "InterviewRepo",
    "PageResult",
    "get_record_manager",
]
