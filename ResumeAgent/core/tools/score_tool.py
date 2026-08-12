"""
core/tools/score_tool.py - 分数计算格式化、雷达图数据构造

功能:
    1. 加权综合得分计算
    2. 分数 → 等级映射
    3. 人类可读的匹配报告格式化
    4. Plotly 雷达图数据结构构造
    5. 优化前后分数对比
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger

from core.graph.agent_state import MatchScoreDetail


# ============================================================
# 常量配置
# ============================================================

# 三维度评分默认权重（总和 = 1.0）
DEFAULT_WEIGHTS: Dict[str, float] = {
    "skill_match": 0.40,
    "experience_match": 0.45,
    "education_match": 0.15,
}

# 分数 → 等级映射
GRADE_THRESHOLDS: List[Tuple[int, int, str, str]] = [
    # (min, max, 等级, 描述)
    (90, 100, "S", "卓越匹配"),
    (80, 89, "A", "高度匹配"),
    (65, 79, "B", "良好匹配"),
    (50, 64, "C", "一般匹配"),
    (0, 49, "D", "待提升"),
]

# 等级对应颜色（Plotly 风格）
GRADE_COLORS: Dict[str, str] = {
    "S": "#00C853",  # 鲜绿
    "A": "#64DD17",  # 浅绿
    "B": "#FFD600",  # 黄
    "C": "#FF9100",  # 橙
    "D": "#FF1744",  # 红
}

# 雷达图三维度标签
RADAR_LABELS: List[str] = ["技能匹配", "经验匹配", "学历匹配"]


# ============================================================
# 分数计算
# ============================================================

def calculate_weighted_score(
    scores: MatchScoreDetail,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    按照权重计算加权综合得分。

    Args:
        scores: MatchScoreDetail 实例
        weights: 自定义权重字典，为 None 时使用 DEFAULT_WEIGHTS

    Returns:
        加权综合得分 (0-100)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    weighted = (
        scores.skill_match * weights.get("skill_match", 0.40)
        + scores.experience_match * weights.get("experience_match", 0.45)
        + scores.education_match * weights.get("education_match", 0.15)
    )

    return round(weighted, 1)


def score_to_grade(score: float) -> Tuple[str, str, str]:
    """
    将分数转换为等级。

    Args:
        score: 0-100 的数值分数

    Returns:
        (等级字母, 等级描述, 对应颜色)
    """
    score = max(0, min(100, score))
    for lo, hi, grade, desc in GRADE_THRESHOLDS:
        if lo <= score <= hi:
            return grade, desc, GRADE_COLORS[grade]
    return "C", "一般匹配", GRADE_COLORS["C"]


def compare_scores(
    before: MatchScoreDetail,
    after: MatchScoreDetail,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    对比优化前后分数。

    Args:
        before: 优化前的 MatchScoreDetail
        after:  优化后的 MatchScoreDetail

    Returns:
        对比结果字典，包含各项变化量
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    before_overall = calculate_weighted_score(before, weights)
    after_overall = calculate_weighted_score(after, weights)

    def _delta(new_val: float, old_val: float) -> float:
        return round(new_val - old_val, 1)

    return {
        "overall": {
            "before": before_overall,
            "after": after_overall,
            "delta": _delta(after_overall, before_overall),
            "grade_before": score_to_grade(before_overall)[:2],
            "grade_after": score_to_grade(after_overall)[:2],
        },
        "skill_match": {
            "before": before.skill_match,
            "after": after.skill_match,
            "delta": _delta(after.skill_match, before.skill_match),
        },
        "experience_match": {
            "before": before.experience_match,
            "after": after.experience_match,
            "delta": _delta(after.experience_match, before.experience_match),
        },
        "education_match": {
            "before": before.education_match,
            "after": after.education_match,
            "delta": _delta(after.education_match, before.education_match),
        },
    }


# ============================================================
# 格式化输出
# ============================================================

def format_match_report(
    scores: MatchScoreDetail,
    weights: Optional[Dict[str, float]] = None,
    title: str = "简历匹配分析报告",
) -> str:
    """
    将 MatchScoreDetail 格式化为人类可读的文本报告。

    Args:
        scores: MatchScoreDetail 实例
        weights: 自定义权重
        title: 报告标题

    Returns:
        格式化后的文本报告
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    overall = calculate_weighted_score(scores, weights)
    grade, desc, _ = score_to_grade(overall)

    def _bar(score_val: float, width: int = 20) -> str:
        filled = int(score_val / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        "",
        f"  综合得分: {overall} / 100  |  等级: {grade}  |  {desc}",
        f"  综合进度: {_bar(overall)}",
        "",
        "-" * 60,
        "  【三维评估明细】",
        "-" * 60,
        f"  技能匹配度:      {scores.skill_match:>5.1f} / 100  {_bar(scores.skill_match)}  (权重 {weights['skill_match']*100:.0f}%)",
        f"  经验匹配度:      {scores.experience_match:>5.1f} / 100  {_bar(scores.experience_match)}  (权重 {weights['experience_match']*100:.0f}%)",
        f"  学历匹配度:      {scores.education_match:>5.1f} / 100  {_bar(scores.education_match)}  (权重 {weights['education_match']*100:.0f}%)",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


def format_compare_report(compare: Dict[str, Any], title: str = "优化前后对比报告") -> str:
    """
    格式化优化前后对比报告。

    Args:
        compare: compare_scores() 的返回值
        title: 报告标题

    Returns:
        格式化后的文本报告
    """
    arrows = {
        True: " ↑",
        False: " ↓",
    }

    def _sign(delta: float) -> str:
        if delta > 0:
            return f"+{delta}"
        return str(delta)

    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        "",
        f"  综合得分: {compare['overall']['before']} → {compare['overall']['after']}  "
        f"({_sign(compare['overall']['delta'])} | "
        f"{compare['overall']['grade_before'][0]} → {compare['overall']['grade_after'][0]})",
        "",
        "-" * 60,
        "  【维度变化】",
        "-" * 60,
    ]

    for key, label in [
        ("skill_match", "技能匹配度"),
        ("experience_match", "经验匹配度"),
        ("education_match", "学历匹配度"),
    ]:
        d = compare[key]
        arrow = arrows.get(d["delta"] >= 0, " →")
        lines.append(
            f"  {label}:  {d['before']} → {d['after']}  ({_sign(d['delta'])}){arrow}"
        )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# 雷达图数据构造
# ============================================================

def build_radar_chart_data(
    scores: MatchScoreDetail,
    labels: Optional[List[str]] = None,
    max_value: int = 100,
) -> Dict[str, Any]:
    """
    构造 Plotly 雷达图所需的 trace 数据。

    Args:
        scores:  MatchScoreDetail 实例
        labels:  维度标签列表，为 None 时使用 RADAR_LABELS
        max_value: 雷达图最大值

    Returns:
        Plotly go.Scatterpolar 兼容的数据字典
    """
    if labels is None:
        labels = RADAR_LABELS

    values = [
        scores.skill_match,
        scores.experience_match,
        scores.education_match,
    ]

    # 填充环形数据（Plotly 雷达图需要闭合）
    fill_values = values + [values[0]]
    fill_labels = labels + [labels[0]]

    return {
        "type": "scatterpolar",
        "r": fill_values,
        "theta": fill_labels,
        "fill": "toself",
        "mode": "markers+lines",
        "marker": {"size": 8, "color": "#4A90D9"},
        "line": {"color": "#4A90D9", "width": 2},
        "fillcolor": "rgba(74, 144, 217, 0.25)",
    }


def build_radar_comparison_data(
    before: MatchScoreDetail,
    after: MatchScoreDetail,
    labels: Optional[List[str]] = None,
    max_value: int = 100,
) -> List[Dict[str, Any]]:
    """
    构造优化前后对比的雷达图数据（两条 trace）。

    Args:
        before: 优化前分数
        after:  优化后分数
        labels: 维度标签
        max_value: 雷达图最大值

    Returns:
        Plotly 双 trace 数据列表
    """
    if labels is None:
        labels = RADAR_LABELS

    before_values = [
        before.skill_match,
        before.experience_match,
        before.education_match,
    ]
    after_values = [
        after.skill_match,
        after.experience_match,
        after.education_match,
    ]

    fill_labels = labels + [labels[0]]

    trace_before = {
        "type": "scatterpolar",
        "name": "优化前",
        "r": before_values + [before_values[0]],
        "theta": fill_labels,
        "fill": "toself",
        "mode": "markers+lines",
        "marker": {"size": 6, "color": "#FF7043"},
        "line": {"color": "#FF7043", "width": 2, "dash": "dash"},
        "fillcolor": "rgba(255, 112, 67, 0.15)",
    }

    trace_after = {
        "type": "scatterpolar",
        "name": "优化后",
        "r": after_values + [after_values[0]],
        "theta": fill_labels,
        "fill": "toself",
        "mode": "markers+lines",
        "marker": {"size": 6, "color": "#4A90D9"},
        "line": {"color": "#4A90D9", "width": 2},
        "fillcolor": "rgba(74, 144, 217, 0.25)",
    }

    return [trace_before, trace_after]


def build_plotly_figure_config(
    title: str = "简历匹配雷达图",
    max_value: int = 100,
) -> Dict[str, Any]:
    """
    构造 Plotly 雷达图布局配置。

    Args:
        title:    图表标题
        max_value: 雷达图最大值

    Returns:
        Plotly layout 配置字典
    """
    return {
        "title": {"text": title, "font": {"size": 18, "color": "#333"}},
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, max_value],
                "showline": False,
                "gridcolor": "#E0E0E0",
            },
            "angularaxis": {
                "gridcolor": "#E0E0E0",
                "linecolor": "#BDBDBD",
            },
        },
        "showlegend": True,
        "legend": {"x": 0.85, "y": 0.1},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
    }


# ============================================================
# 辅助工具
# ============================================================

def format_score_bar(score: float, width: int = 15, label: Optional[str] = None) -> str:
    """
    生成单个分数的进度条字符串，方便前端展示。

    Args:
        score: 0-100 分数
        width: 进度条宽度（字符数）
        label: 维度名称

    Returns:
        进度条字符串，如 "技能匹配: ████████░░░░░░░░  55/100"
    """
    filled = int(max(0, min(score, 100)) / 100 * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    prefix = f"{label}: " if label else ""
    return f"{prefix}{bar}  {score:.0f}/100"


def scores_to_dict(scores: MatchScoreDetail) -> Dict[str, float]:
    """
    将 MatchScoreDetail 提取为纯字典，便于 JSON 序列化。

    Args:
        scores: MatchScoreDetail 实例

    Returns:
        四维分数字典
    """
    return {
        "overall_score": scores.overall_score,
        "skill_match": scores.skill_match,
        "experience_match": scores.experience_match,
        "education_match": scores.education_match,
    }
