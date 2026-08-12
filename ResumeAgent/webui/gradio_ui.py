"""
webui/gradio_ui.py - Gradio 六 Tab 前端界面

Tab1: 知识库管理 - 上传 PDF/Word 入库、查看文档、清空库、检索测试、统计图表
Tab2: 简历 & JD 匹配分析 - 上传解析、雷达图/柱状图、缺失技能清单
Tab3: 简历智能优化 - 左右分栏对比、差异高亮、优化模式、下载 Word
Tab4: AI 模拟面试 - 聊天对话界面、流式输出、暂停/重置、复盘报告下载
Tab5: 历史记录 - 查看匹配/简历/面试记录
Tab6: 自主 Agent - 输入自然语言任务，ReActAgent 自主调用工具完成（v2.0）
"""

import difflib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
from loguru import logger

from config.settings import PROJECT_ROOT, settings
from core.graph.agent_nodes import (
    interview_generate_question,
    interview_evaluate_answer,
    optimize_node,
    parse_node,
    retrieve_node,
    score_node,
    summary_node,
)
from core.graph.agent_state import AgentState, MatchScoreDetail
from core.rag.document_loader import DocumentLoader
from core.rag.text_splitter import DocumentSplitter
from core.rag.vector_store import VectorStoreManager
from core.tools.file_export import (
    export_interview_report_to_txt,
    export_match_report_to_txt,
    export_resume_to_docx,
)
from core.tools.score_tool import (
    RADAR_LABELS,
    build_radar_chart_data,
    build_radar_comparison_data,
    calculate_weighted_score,
    score_to_grade,
    scores_to_dict,
)
from core.tools.sqlite_db import RecordManager
from core.graph.workflow_graph import run_agentic_task

# ============================================================
# 辅助函数
# ============================================================

def _parse_uploaded_file(file_obj) -> Tuple[str, str]:
    """解析上传文件，返回 (文本内容, 文件名)"""
    if file_obj is None:
        return "", ""
    try:
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)
        loader = DocumentLoader()
        text = loader.load(file_path)  # 直接返回清洗后纯文本
        file_name = Path(file_path).name
        return text, file_name
    except Exception as e:
        logger.error(f"文件解析失败: {e}")
        return f"[解析失败] {str(e)}", ""


def _compute_diff_html(original: str, optimized: str) -> Tuple[str, str]:
    """
    计算原文与优化后的差异，生成 HTML 高亮。
    返回 (原文高亮HTML, 优化后高亮HTML)
    """
    orig_words = re.findall(r'\S+|\s+', original)
    opt_words = re.findall(r'\S+|\s+', optimized)

    matcher = difflib.SequenceMatcher(None, orig_words, opt_words)

    orig_html_parts = []
    opt_html_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            orig_html_parts.append(''.join(orig_words[i1:i2]))
            opt_html_parts.append(''.join(opt_words[j1:j2]))
        elif tag == 'replace':
            orig_html_parts.append(
                f'<span style="background-color:#ffcccb;text-decoration:line-through;">'
                f'{"".join(orig_words[i1:i2])}</span>'
            )
            opt_html_parts.append(
                f'<span style="background-color:#90ee90;">'
                f'{"".join(opt_words[j1:j2])}</span>'
            )
        elif tag == 'delete':
            orig_html_parts.append(
                f'<span style="background-color:#ffcccb;text-decoration:line-through;">'
                f'{"".join(orig_words[i1:i2])}</span>'
            )
        elif tag == 'insert':
            opt_html_parts.append(
                f'<span style="background-color:#90ee90;">'
                f'{"".join(opt_words[j1:j2])}</span>'
            )

    orig_html = ''.join(orig_html_parts).replace('\n', '<br>')
    opt_html = ''.join(opt_html_parts).replace('\n', '<br>')

    return f'<div style="font-family:monospace;white-space:pre-wrap;">{orig_html}</div>', \
           f'<div style="font-family:monospace;white-space:pre-wrap;">{opt_html}</div>'


# ============================================================
# Tab1: 知识库管理
# ============================================================

def _kb_upload_file(file_obj) -> str:
    """上传文件到向量知识库"""
    if file_obj is None:
        return "⚠️ 请先选择文件"
    text, fname = _parse_uploaded_file(file_obj)
    if not text.strip() or text.startswith("[解析失败]"):
        return f"❌ 文件解析失败: {text}"
    try:
        splitter = DocumentSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        docs = splitter.split(text, metadata={"source": fname})
        vs = VectorStoreManager()
        ids = vs.add_documents(docs)
        return f"✅ 入库成功！文件 `{fname}` 已切分为 {len(ids)} 个片段存入向量库"
    except Exception as e:
        logger.error(f"知识库入库失败: {e}")
        return f"❌ 入库失败: {str(e)}"


def _kb_get_stats() -> str:
    """获取知识库统计信息"""
    try:
        vs = VectorStoreManager()
        info = vs.get_collection_info()
        return (
            f"### 知识库统计\n"
            f"- **Collection**: `{info['name']}`\n"
            f"- **文档片段总数**: {info['document_count']}\n"
            f"- **持久化路径**: `{info['persist_dir']}`\n"
        )
    except Exception as e:
        return f"❌ 获取统计失败: {str(e)}"


def _kb_list_docs() -> str:
    """列出向量库中的文档来源"""
    try:
        vs = VectorStoreManager()
        count = vs.count
        if count == 0:
            return "知识库为空，请先上传文档。"
        # 通过 get 方法获取部分文档元数据
        collection = vs.vector_store._collection
        results = collection.get(limit=min(count, 100), include=["metadatas"])
        metadatas = results.get("metadatas", [])
        # 按来源分组统计
        source_counts: Dict[str, int] = {}
        for m in metadatas:
            src = m.get("source", "unknown") if m else "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

        lines = ["### 知识库文档清单\n", "| 来源文件 | 片段数 |", "|---------|--------|"]
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {src} | {cnt} |")
        lines.append(f"\n共 **{len(source_counts)}** 个文件，**{count}** 个片段")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 获取文档列表失败: {str(e)}"


def _kb_clear() -> str:
    """清空向量知识库"""
    try:
        vs = VectorStoreManager()
        before = vs.count
        vs.clear()
        return f"🗑️ 知识库已清空（原 {before} 个文档片段已删除）"
    except Exception as e:
        return f"❌ 清空失败: {str(e)}"


def _kb_test_search(query: str, top_k: int) -> str:
    """测试知识库检索"""
    if not query.strip():
        return "⚠️ 请输入检索关键词"
    try:
        vs = VectorStoreManager()
        results = vs.similarity_search_with_score(query, k=top_k)
        if not results:
            return "未检索到相关内容。"
        lines = [f"### 检索结果 (Top-{top_k}: \"{query}\")\n"]
        for i, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            text_preview = doc.page_content[:200].replace("\n", " ")
            lines.append(
                f"**#{i}** [{src}] 相似度: `{score:.4f}`\n"
                f"> {text_preview}...\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 检索失败: {str(e)}"


def _kb_stats_chart() -> go.Figure:
    """知识库来源分布饼图"""
    try:
        vs = VectorStoreManager()
        if vs.count == 0:
            fig = go.Figure()
            fig.add_annotation(text="暂无数据", showarrow=False, font=dict(size=20))
            fig.update_layout(height=350)
            return fig
        collection = vs.vector_store._collection
        results = collection.get(limit=min(vs.count, 500), include=["metadatas"])
        metadatas = results.get("metadatas", [])
        source_counts: Dict[str, int] = {}
        for m in metadatas:
            src = m.get("source", "unknown") if m else "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

        labels = list(source_counts.keys())
        values = list(source_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4,
                                      textinfo='label+value')])
        fig.update_layout(title="知识库来源分布", height=400)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"加载失败: {e}", showarrow=False)
        return fig


def build_tab1_knowledge_base() -> None:
    """构建 Tab1：知识库管理"""
    gr.Markdown("## 📚 知识库管理")
    gr.Markdown("上传简历范文、面试经验、岗位要求等文档，构建专属知识库辅助匹配与面试。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 文档入库")
            upload_file = gr.File(
                label="上传 PDF/Word 文档",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
            )
            btn_upload = gr.Button("📤 一键入库", variant="primary")
            upload_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### 检索测试")
            search_query = gr.Textbox(label="检索关键词", placeholder="如：Python 后端开发")
            search_topk = gr.Slider(1, 10, value=3, step=1, label="返回数量 Top-K")
            btn_search = gr.Button("🔍 测试检索")
            search_result = gr.Markdown("")

        with gr.Column(scale=1):
            stats_md = gr.Markdown("")
            btn_refresh = gr.Button("🔄 刷新统计")
            doc_list_md = gr.Markdown("")
            btn_list = gr.Button("📋 查看文档清单")
            btn_clear = gr.Button("🗑️ 清空知识库", variant="stop")
            clear_status = gr.Markdown("")

    gr.Markdown("---")
    stats_chart = gr.Plot(label="知识库来源分布")

    # 事件绑定
    btn_upload.click(_kb_upload_file, inputs=[upload_file], outputs=[upload_status])
    btn_refresh.click(_kb_get_stats, outputs=[stats_md])
    btn_list.click(_kb_list_docs, outputs=[doc_list_md])
    btn_clear.click(_kb_clear, outputs=[clear_status]).then(
        _kb_get_stats, outputs=[stats_md]
    ).then(_kb_list_docs, outputs=[doc_list_md]).then(
        _kb_stats_chart, outputs=[stats_chart]
    )
    btn_search.click(
        _kb_test_search, inputs=[search_query, search_topk], outputs=[search_result]
    )
    # 初始加载
    stats_md.value = _kb_get_stats()
    doc_list_md.value = _kb_list_docs()
    stats_chart.value = _kb_stats_chart()


# ============================================================
# Tab2: 简历 & JD 匹配分析
# ============================================================

def _match_analyze(resume_file, jd_text: str) -> Tuple[
    go.Figure, go.Figure, str, str, str, str, str, str, str, str
]:
    """
    执行简历-JD匹配分析，返回图表和数据。
    """
    if resume_file is None:
        empty_fig = _empty_chart("请上传简历文件")
        return (empty_fig, empty_fig, "", "", "", "", "", "请上传简历文件", "", "")

    resume_text, resume_fname = _parse_uploaded_file(resume_file)
    if not resume_text.strip():
        empty_fig = _empty_chart("简历解析为空")
        return (empty_fig, empty_fig, "", "", "", "", "", "简历解析失败", "", "")

    if not jd_text.strip():
        empty_fig = _empty_chart("请输入或上传JD内容")
        return (empty_fig, empty_fig, "", "", "", "", "", "请填写JD内容", "", "")

    try:
        # 构建 AgentState Pydantic 实例（节点函数需要 attribute access）
        state = AgentState(resume_raw=resume_text, jd_raw=jd_text)

        # Step1: 解析
        parsed = parse_node(state)
        state = state.model_copy(update=parsed)

        # Step2: 知识库检索
        retrieved = retrieve_node(state)
        state = state.model_copy(update=retrieved)

        # Step3: 评分
        scored = score_node(state)
        state = state.model_copy(update=scored)

        score_detail = state.match_score
        skill_gap = state.skill_gap or []

        if score_detail is None:
            empty_fig = _empty_chart("评分生成失败")
            return (empty_fig, empty_fig, state.resume_raw[:500], state.jd_raw[:500],
                    "", "", "", "评分生成失败，请重试", "", "")

        # 雷达图
        radar_fig = _build_radar_fig(score_detail)

        # 柱状图
        bar_fig = _build_bar_fig(score_detail)

        # 文本报告
        match_report = _format_match_md(score_detail, skill_gap)

        # 等级
        grade = score_to_grade(score_detail.overall_score)

        return (
            radar_fig, bar_fig,
            state.resume_raw[:2000], state.jd_raw[:2000],
            match_report,
            f"### 综合评分: {score_detail.overall_score:.1f} / 100  ({grade} 级)",
            f"技能匹配: {score_detail.skill_match:.1f} | 经验匹配: {score_detail.experience_match:.1f} | 学历匹配: {score_detail.education_match:.1f}",
            _format_skill_gap_table(skill_gap),
            resume_fname,
            jd_text[:500],
        )

    except Exception as e:
        logger.error(f"匹配分析失败: {e}")
        import traceback
        traceback.print_exc()
        empty_fig = _empty_chart(f"分析异常: {str(e)}")
        return (empty_fig, empty_fig, "", "", "", "", "", f"❌ 分析失败: {e}", "", "")


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=18, color="gray"))
    fig.update_layout(height=350)
    return fig


def _build_radar_fig(score_detail: MatchScoreDetail) -> go.Figure:
    chart_data = build_radar_chart_data(score_detail)
    fig = go.Figure(data=go.Scatterpolar(
        r=chart_data["r"], theta=chart_data["theta"],
        fill='toself', name='匹配度',
        line=dict(color='rgba(59, 130, 246, 0.8)', width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="技能-经验-学历三维度雷达图",
        height=400,
    )
    return fig


def _build_bar_fig(score_detail: MatchScoreDetail) -> go.Figure:
    categories = list(RADAR_LABELS) + ["综合评分"]
    values = [
        score_detail.skill_match,
        score_detail.experience_match,
        score_detail.education_match,
        score_detail.overall_score,
    ]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

    fig = go.Figure(data=[go.Bar(
        x=categories, y=values, marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition='outside',
    )])
    fig.update_layout(
        title="匹配度柱状图",
        yaxis=dict(range=[0, 105], title="分数"),
        height=400,
    )
    return fig


def _format_match_md(score_detail: MatchScoreDetail, skill_gap: List[str]) -> str:
    """格式化匹配报告 MD"""
    lines = [
        "### 匹配分析报告",
        "",
        "| 维度 | 得分 | 权重 | 加权贡献 |",
        "|------|------|------|----------|",
    ]
    weights = {"skill_match": 0.40, "experience_match": 0.45, "education_match": 0.15}
    for key, label in [("skill_match", "技能匹配"), ("experience_match", "经验匹配"), ("education_match", "学历匹配")]:
        s = getattr(score_detail, key, 0)
        w = weights[key]
        lines.append(f"| {label} | {s:.1f} | {w*100:.0f}% | {s*w:.1f} |")

    overall = score_detail.overall_score
    grade = score_to_grade(overall)
    lines.append(f"\n**综合得分**: {overall:.1f} / 100  **等级**: {grade}")

    if skill_gap:
        lines.append(f"\n**缺失技能**: {', '.join(skill_gap[:10])}")
    return "\n".join(lines)


def _format_skill_gap_table(skill_gap: List[str]) -> str:
    if not skill_gap:
        return "### 缺失技能清单\n\n✅ 无显著技能缺口"
    lines = ["### 缺失技能清单\n", "| # | 技能 |", "|---|------|"]
    for i, skill in enumerate(skill_gap[:20], 1):
        lines.append(f"| {i} | {skill} |")
    return "\n".join(lines)


def build_tab2_match_analysis() -> None:
    """构建 Tab2：简历 & JD 匹配分析"""
    gr.Markdown("## 📊 简历 & JD 匹配分析")
    gr.Markdown("上传简历和岗位描述，一键获得多维匹配分析和缺失技能清单。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 文件上传")
            resume_upload = gr.File(
                label="上传简历 (PDF/Word)",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
            )
            jd_input = gr.Textbox(
                label="岗位描述 (JD) - 直接粘贴或输入",
                placeholder="在此粘贴/输入 JD 内容...",
                lines=8,
            )
            btn_analyze = gr.Button("🔬 开始匹配分析", variant="primary", size="lg")

    with gr.Row():
        match_summary = gr.Markdown("")
        match_detail = gr.Markdown("")

    gr.Markdown("---")

    with gr.Row():
        radar_chart = gr.Plot(label="三维度雷达图")
        bar_chart = gr.Plot(label="匹配度柱状图")

    gr.Markdown("---")
    with gr.Row():
        match_report_md = gr.Markdown("")
        skill_gap_table = gr.Markdown("")

    with gr.Accordion("📄 解析预览", open=False):
        with gr.Row():
            resume_preview = gr.Textbox(label="简历解析内容", lines=12, max_lines=20)
            jd_preview = gr.Textbox(label="JD解析内容", lines=12, max_lines=20)

    def _on_analyze(resume_file, jd_text):
        results = _match_analyze(resume_file, jd_text)
        # _match_analyze returns: [0]radar [1]bar [2]resume_pv [3]jd_pv [4]match_report
        #                         [5]summary [6]dims [7]gap_table [8]fname [9]jd_raw
        return (
            results[0],  # radar_chart
            results[1],  # bar_chart
            results[4],  # match_report_md
            results[5],  # match_summary
            results[6],  # match_detail
            results[7],  # skill_gap_table
            results[2],  # resume_preview
            results[3],  # jd_preview
        )

    # Output mapping: radar, bar, match_summary, match_detail, skill_gap, resume_pv, jd_pv
    btn_analyze.click(
        _on_analyze,
        inputs=[resume_upload, jd_input],
        outputs=[
            radar_chart, bar_chart,
            match_report_md,
            match_summary,
            match_detail,
            skill_gap_table,
            resume_preview, jd_preview,
        ],
    )


# ============================================================
# Tab3: 简历智能优化
# ============================================================

def _optimize_resume(
    resume_file, jd_text: str, mode: str, feedback: str, iteration: int,
    prev_optimized: str = "",
) -> Tuple[str, str, str, str, str, Optional[str], int, str]:
    """
    执行简历优化，返回对比和下载路径。

    Args:
        resume_file: 上传的简历文件
        jd_text: 岗位 JD 文本
        mode: "keywords" | "quantify" | "concise"
        feedback: 用户额外优化要求
        iteration: 当前迭代次数
        prev_optimized: 上一轮优化结果（多轮迭代时传入）

    Returns:
        (orig_html, opt_html, score_info, opt_text, status, download_path, new_iteration, optimized_resume)
    """
    if resume_file is None:
        return "", "", "请上传简历文件", "", "缺少简历", None, iteration, ""

    resume_text, fname = _parse_uploaded_file(resume_file)
    if not resume_text.strip():
        return "", "", "简历解析为空", "", "简历解析失败", None, iteration, ""

    if not jd_text.strip():
        return "", "", "请填写JD内容", "", "缺少JD", None, iteration, ""

    try:
        # 构建 AgentState，传入优化模式和用户反馈
        state = AgentState(
            resume_raw=resume_text,
            jd_raw=jd_text,
            optimize_mode=mode or "keywords",
            optimize_feedback=feedback.strip(),
        )

        # 多轮迭代：继承上一轮优化结果，使 optimize_node 能在此基础上继续优化
        if iteration > 1 and prev_optimized.strip():
            state.optimized_resume = prev_optimized
            logger.info(f"  多轮迭代 #{iteration}：基于上一版 ({len(prev_optimized)} 字符) 继续优化")

        # 解析
        parsed = parse_node(state)
        state = state.model_copy(update=parsed)

        # 检索
        retrieved = retrieve_node(state)
        state = state.model_copy(update=retrieved)

        # 评分（优化前 — 基于原始简历）
        scored = score_node(state)
        state = state.model_copy(update=scored)
        before_score = state.match_score.overall_score if state.match_score else 0
        before_grade = score_to_grade(before_score) if before_score else "N/A"

        # 优化
        opt_result = optimize_node(state)
        state = state.model_copy(update=opt_result)

        optimized_resume = state.optimized_resume or state.resume_raw
        original_resume = state.resume_raw

        # 生成差异高亮 HTML
        orig_html, opt_html = _compute_diff_html(original_resume, optimized_resume)

        # 优化后评分 — 用优化后的简历文本临时替换 resume_raw 以正确评分
        # （score_node 基于 resume_raw vs jd_raw 评分）
        saved_raw = state.resume_raw
        state.resume_raw = optimized_resume
        scored2 = score_node(state)
        state = state.model_copy(update=scored2)
        state.resume_raw = saved_raw  # 恢复
        after_score = state.match_score.overall_score if state.match_score else 0
        after_grade = score_to_grade(after_score) if after_score else "N/A"

        # 生成下载文件
        exports_dir = PROJECT_ROOT / "exports"
        os.makedirs(str(exports_dir), exist_ok=True)
        output_path = str(exports_dir / f"optimized_{fname}.docx")
        try:
            export_resume_to_docx(optimized_resume, output_path=output_path)
        except Exception as e:
            logger.warning(f"docx 导出失败，使用 txt 作为降级: {e}")
            output_path = str(exports_dir / f"optimized_{fname}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(optimized_resume)

        improvement = after_score - before_score
        score_info = (
            f"**优化前**: {before_score:.1f} ({before_grade}) → "
            f"**优化后**: {after_score:.1f} ({after_grade}) | "
            f"**提升**: {'+' if improvement >= 0 else ''}{improvement:.1f} 分"
        )

        return orig_html, opt_html, score_info, optimized_resume[:3000], "", output_path, iteration + 1, optimized_resume

    except Exception as e:
        logger.error(f"简历优化失败: {e}")
        import traceback
        traceback.print_exc()
        return "", "", f"❌ 优化失败: {e}", "", f"❌ 错误: {e}", None, iteration, ""


def build_tab3_optimization() -> None:
    """构建 Tab3：简历智能优化"""
    gr.Markdown("## ✨ 简历智能优化")
    gr.Markdown("上传简历和对应 JD，选择优化模式，左右对比查看差异，一键下载优化版简历。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 输入")
            resume_upload = gr.File(
                label="上传简历 (PDF/Word)",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
            )
            jd_input = gr.Textbox(
                label="岗位描述 (JD)",
                placeholder="在此粘贴/输入目标岗位 JD 内容...",
                lines=6,
            )
            opt_mode = gr.Radio(
                choices=["keywords", "quantify", "concise"],
                value="keywords",
                label="优化模式",
                info="keywords=侧重关键词匹配 | quantify=侧重项目量化 | concise=精简冗余",
            )
            opt_feedback = gr.Textbox(
                label="额外优化要求（选填）",
                placeholder="如：突出大数据处理经验、缩短到一页...",
                lines=2,
            )
            iter_state = gr.State(1)
            prev_optimized_state = gr.State("")  # 存储上一轮优化结果，用于多轮迭代
            with gr.Row():
                btn_optimize = gr.Button("🚀 开始优化", variant="primary")
                btn_re_optimize = gr.Button("🔄 不满意，重新优化", variant="secondary")

    gr.Markdown("---")

    score_display = gr.Markdown("")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 优化前")
            orig_html = gr.HTML(label="原文")
        with gr.Column():
            gr.Markdown("### ✨ 优化后")
            opt_html = gr.HTML(label="优化版")

    with gr.Row():
        download_btn = gr.DownloadButton(
            label="📥 下载优化简历 (Word)", variant="primary", visible=False
        )
        opt_full_text = gr.Textbox(label="优化后纯文本", lines=15, max_lines=30, visible=False)

    status_md = gr.Markdown("")

    def _do_optimize(resume_file, jd_text, mode, feedback, it, prev_opt):
        orig_h, opt_h, score_info, opt_text, status, path, new_it, full_opt = _optimize_resume(
            resume_file, jd_text, mode, feedback, it, prev_opt
        )
        return orig_h, opt_h, score_info, opt_text, status, path, new_it, full_opt

    btn_optimize.click(
        _do_optimize,
        inputs=[resume_upload, jd_input, opt_mode, opt_feedback, iter_state, prev_optimized_state],
        outputs=[orig_html, opt_html, score_display, opt_full_text, status_md, download_btn, iter_state, prev_optimized_state],
    ).then(
        lambda p: gr.update(visible=p is not None),
        inputs=[download_btn],
        outputs=[download_btn],
    )

    btn_re_optimize.click(
        _do_optimize,
        inputs=[resume_upload, jd_input, opt_mode, opt_feedback, iter_state, prev_optimized_state],
        outputs=[orig_html, opt_html, score_display, opt_full_text, status_md, download_btn, iter_state, prev_optimized_state],
    ).then(
        lambda p: gr.update(visible=p is not None),
        inputs=[download_btn],
        outputs=[download_btn],
    )


# ============================================================
# Tab4: AI 模拟面试
# ============================================================

def _interview_start(jd_text: str, resume_text: str, num_questions: int) -> Tuple[
    List[Tuple[str, str]], Dict[str, Any], str
]:
    """初始化面试，生成第一个问题"""
    if not jd_text.strip():
        return [], {}, "请填写 JD 内容"
    try:
        # 构建 AgentState Pydantic 实例
        resume_raw = resume_text if resume_text.strip() else "未提供简历"
        state = AgentState(jd_raw=jd_text, resume_raw=resume_raw)

        # 解析
        parsed = parse_node(state)
        state = state.model_copy(update=parsed)

        # 检索 RAG 资料
        try:
            retrieved = retrieve_node(state)
            state = state.model_copy(update=retrieved)
        except Exception:
            pass

        # 生成第一道面试题
        result = interview_generate_question(state)
        state = state.model_copy(update=result)

        # 从 interview_history 获取问题
        history = state.interview_history
        if not history or len(history) == 0:
            return [], {}, "未能生成面试题目，请重试"

        first_qa = history[0]
        first_q = first_qa.question

        welcome = f"面试开始！共有 {num_questions} 道题。\n\n**第 1 题**: {first_q}"
        chat_history = [{"role": "assistant", "content": welcome}]

        session_state = {
            "state_model": state,   # AgentState Pydantic 实例
            "questions": [],         # 将按需生成
            "total_rounds": num_questions,
            "current_round": 1,
            "answers": [],
            "jd_text": jd_text,
            "active": True,
            "paused": False,
        }
        return chat_history, session_state, "面试已开始"

    except Exception as e:
        logger.error(f"面试初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, f"初始化失败: {e}"


def _interview_chat(
    user_msg: str, chat_history: List[Tuple[str, str]], session_state: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    """
    处理用户回答 → 评估 → 生成下一题
    """
    if not session_state or not session_state.get("active"):
        chat_history.append({"role": "assistant", "content": "面试尚未开始或已结束，请点击「开始面试」。"})
        return chat_history, session_state, ""

    if session_state.get("paused"):
        chat_history.append({"role": "assistant", "content": "面试已暂停，点击「暂停/继续」恢复。"})
        return chat_history, session_state, ""

    state: AgentState = session_state["state_model"]
    current_round = session_state["current_round"]
    total_rounds = session_state["total_rounds"]

    try:
        # 1. 评估当前回答（当前轮次在 interview_history 中的索引是 current_round - 1）
        round_index = current_round - 1
        current_qa = state.interview_history[round_index]
        current_q = current_qa.question

        eval_result = interview_evaluate_answer(state, round_index, user_msg)
        state = state.model_copy(update=eval_result)
        session_state["state_model"] = state

        # 获取评估结果
        updated_qa = state.interview_history[round_index]
        eval_score = updated_qa.score
        eval_feedback = updated_qa.feedback

        # 记录答案
        session_state["answers"].append({
            "question": current_q,
            "answer": user_msg,
            "score": eval_score,
            "feedback": eval_feedback,
        })

        chat_history.append({"role": "user", "content": user_msg})
        feedback_line = f"评分: {eval_score}/100\n{eval_feedback}"
        chat_history.append({"role": "assistant", "content": feedback_line})

        # 2. 判断是否结束
        if current_round >= total_rounds:
            chat_history.append({"role": "assistant", "content": "所有题目已完成！点击「结束面试」生成复盘报告。"})
            session_state["active"] = False
            return chat_history, session_state, ""

        # 3. 生成下一题
        current_round += 1
        session_state["current_round"] = current_round

        gen_result = interview_generate_question(state)
        state = state.model_copy(update=gen_result)
        session_state["state_model"] = state

        next_qa = state.interview_history[current_round - 1]
        next_q = next_qa.question

        next_msg = f"**第 {current_round}/{total_rounds} 题**: {next_q}"
        chat_history.append({"role": "assistant", "content": next_msg})

        return chat_history, session_state, ""

    except Exception as e:
        logger.error(f"面试对话异常: {e}")
        import traceback
        traceback.print_exc()
        chat_history.append({"role": "assistant", "content": f"系统错误: {e}"})
        return chat_history, session_state, ""


def _interview_end(session_state: Dict[str, Any]) -> Tuple[
    List[Tuple[str, str]], Dict[str, Any], Optional[str], str
]:
    """结束面试，生成复盘报告"""
    if not session_state or not session_state.get("answers"):
        if session_state and session_state.get("state_model"):
            pass  # 有状态但没有回答
        else:
            return [{"role": "assistant", "content": "无面试记录可生成报告。"}], session_state, None, ""

    try:
        state: AgentState = session_state["state_model"]
        answers = session_state.get("answers", [])
        summary_result = summary_node(state)
        state = state.model_copy(update=summary_result)
        session_state["state_model"] = state

        # 生成报告
        total_score = sum(a["score"] for a in answers) / max(len(answers), 1)

        # 构建报告文本
        report_lines = ["# 面试复盘报告\n"]
        report_lines.append(f"**总分**: {total_score:.1f} / 100\n")
        report_lines.append("## 答题详情\n")
        for i, a in enumerate(answers, 1):
            report_lines.append(f"### 第{i}题")
            report_lines.append(f"**题目**: {a['question']}")
            report_lines.append(f"**你的回答**: {a['answer']}")
            report_lines.append(f"**评分**: {a['score']}/100")
            report_lines.append(f"**反馈**: {a['feedback']}\n")

        report_text = "\n".join(report_lines)

        # 保存报告到文件
        import tempfile
        report_path = str(PROJECT_ROOT / "exports" / f"interview_report_{int(time.time())}.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 保存到数据库
        try:
            mgr = RecordManager()
            jd_text = session_state.get("jd_text", "")
            session_id = mgr.interview.create_session(
                job_title="面试练习",
                questions=[{"question": a["question"]} for a in answers],
            )
            for i, a in enumerate(answers):
                mgr.interview.save_answer(
                    session_id, i + 1, a["answer"], a["score"], a["feedback"]
                )
            mgr.interview.complete(session_id, total_score, "面试完成")
        except Exception as e:
            logger.warning(f"面试记录保存失败: {e}")

        chat_history = [{"role": "assistant", "content": f"📋 面试结束！复盘报告已生成。\n\n**总分**: {total_score:.1f}/100\n\n{report_text[:1500]}..."}]
        session_state["active"] = False

        return chat_history, session_state, report_path, ""

    except Exception as e:
        logger.error(f"面试结束异常: {e}")
        return [{"role": "assistant", "content": f"❌ 生成报告失败: {e}"}], session_state, None, ""


def build_tab4_interview() -> None:
    """构建 Tab4：AI 模拟面试"""
    gr.Markdown("## 🎤 AI 模拟面试")
    gr.Markdown("基于 JD 智能生成面试题目，实时评估回答质量，支持暂停/重置，结束后生成复盘报告。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 面试设置")
            jd_text = gr.Textbox(
                label="岗位描述 (JD)",
                placeholder="粘贴目标岗位 JD，面试题将基于此生成...",
                lines=6,
            )
            resume_text = gr.Textbox(
                label="简历内容 (选填)",
                placeholder="可选粘贴简历文本，让面试更具针对性...",
                lines=4,
            )
            num_q = gr.Slider(3, 10, value=5, step=1, label="题目数量")
            with gr.Row():
                btn_start = gr.Button("🎬 开始面试", variant="primary")
                btn_pause = gr.Button("⏸️ 暂停/继续")
                btn_reset = gr.Button("🔄 重置面试", variant="stop")
                btn_end = gr.Button("🏁 结束面试 & 生成报告", variant="secondary")

        with gr.Column(scale=2):
            status_line = gr.Markdown("👆 请先设置 JD 并开始面试")
            chatbot = gr.Chatbot(label="面试对话", height=500)

    with gr.Row():
        user_input = gr.Textbox(
            label="输入你的回答",
            placeholder="在此输入你的回答...",
            lines=3,
            scale=9,
        )
        btn_send = gr.Button("📤 发送", variant="primary", scale=1)

    download_report = gr.DownloadButton(
        label="📥 下载复盘报告", variant="primary", visible=False
    )

    # Session state
    session = gr.State({})

    # ---- 事件绑定 ----

    def _on_start(jd, resume, n):
        history, sess, msg = _interview_start(jd, resume, n)
        return history, sess, msg

    btn_start.click(
        _on_start,
        inputs=[jd_text, resume_text, num_q],
        outputs=[chatbot, session, status_line],
    )

    def _on_send(msg, history, sess):
        if not msg.strip():
            return history, sess, ""
        new_history, new_sess, _ = _interview_chat(msg, history, sess)
        return new_history, new_sess, ""

    btn_send.click(
        _on_send,
        inputs=[user_input, chatbot, session],
        outputs=[chatbot, session, user_input],
    )

    def _on_pause(history, sess):
        if sess and sess.get("active"):
            sess["paused"] = not sess.get("paused", False)
            state_str = "⏸️ 已暂停" if sess["paused"] else "▶️ 已继续"
            history.append({"role": "assistant", "content": state_str})
        return history, sess

    btn_pause.click(
        _on_pause,
        inputs=[chatbot, session],
        outputs=[chatbot, session],
    )

    def _on_reset():
        return [], {}, "🔄 面试已重置，请重新开始"

    btn_reset.click(
        _on_reset,
        outputs=[chatbot, session, status_line],
    ).then(lambda: gr.update(visible=False), outputs=[download_report])

    def _on_end(history, sess):
        if not sess or not sess.get("answers"):
            history.append({"role": "assistant", "content": "⚠️ 尚未进行面试，请先开始。"})
            return history, sess, None
        new_history, new_sess, report_path, _ = _interview_end(sess)
        return new_history, new_sess, report_path

    btn_end.click(
        _on_end,
        inputs=[chatbot, session],
        outputs=[chatbot, session, download_report],
    ).then(
        lambda p: gr.update(visible=p is not None),
        inputs=[download_report],
        outputs=[download_report],
    )


# ============================================================
# Tab5: 历史记录
# ============================================================

def _history_load_all() -> Tuple[str, str, str, str, str]:
    """加载所有历史记录"""
    try:
        mgr = RecordManager()

        # 统计数据
        stats = mgr.statistics()
        stats_md = (
            f"### 系统数据概览\n"
            f"| 类型 | 数量 |\n|------|------|\n"
            f"| 简历 | {stats.get('resume_count', 0)} |\n"
            f"| JD | {stats.get('jd_count', 0)} |\n"
            f"| 优化记录 | {stats.get('optimization_count', 0)} |\n"
            f"| 面试记录 | {stats.get('interview_count', 0)} |\n"
            f"| 面试均分 | {stats.get('avg_interview_score', 0):.1f} |"
        )

        # 简历列表
        resumes = mgr.resume.list_all(limit=20)
        if resumes:
            resume_rows = ["| ID | 姓名 | 创建时间 |", "|----|------|----------|"]
            for r in resumes:
                name = r.get("name", "N/A")
                created = r.get("created_at", "N/A")
                resume_rows.append(f"| {r['id']} | {name} | {created} |")
            resume_md = "\n".join(resume_rows)
        else:
            resume_md = "暂无简历记录"

        # 优化记录
        opts = mgr.optimization.list_all(limit=20)
        if opts:
            opt_rows = ["| ID | 简历 | JD | 优化前 | 优化后 | 时间 |",
                        "|----|------|----|--------|--------|------|"]
            for o in opts:
                resume_name = o.get("resume_name", "N/A")
                job_title = o.get("job_title", "N/A")
                before = o.get("match_score_before", 0)
                after = o.get("match_score_after", 0)
                created = o.get("created_at", "N/A")
                opt_rows.append(
                    f"| {o['id']} | {resume_name} | {job_title} | "
                    f"{before:.1f} | {after:.1f} | {created} |"
                )
            opt_md = "\n".join(opt_rows)
        else:
            opt_md = "暂无优化记录"

        # 面试记录
        sessions = mgr.interview.list_sessions(limit=20)
        if sessions:
            sess_rows = ["| Session ID | 岗位 | 状态 | 总分 | 时间 |",
                         "|------------|------|------|------|------|"]
            for s in sessions:
                sid = s.get("session_id", "N/A")[:20]
                jt = s.get("job_title", "N/A")
                status = s.get("status", "N/A")
                score = s.get("total_score", 0) or 0
                started = s.get("started_at", "N/A")
                sess_rows.append(f"| {sid} | {jt} | {status} | {score:.1f} | {started} |")
            interview_md = "\n".join(sess_rows)
        else:
            interview_md = "暂无面试记录"

        return stats_md, resume_md, opt_md, interview_md, ""

    except Exception as e:
        err = f"❌ 加载历史记录失败: {e}"
        return err, err, err, err, err


def _history_clear() -> str:
    """清空所有历史"""
    try:
        mgr = RecordManager()
        result = mgr.clear_all(confirm=True)
        return f"🗑️ 已清空所有历史数据: {result}"
    except Exception as e:
        return f"❌ 清空失败: {e}"


def _history_export() -> str:
    """导出所有历史"""
    try:
        mgr = RecordManager()
        paths = mgr.export_all()
        lines = ["### 导出成功\n"]
        for name, path in paths.items():
            lines.append(f"- **{name}**: `{path}`")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 导出失败: {e}"


def _history_search(keyword: str, category: str, page: int) -> str:
    """搜索历史记录"""
    if not keyword.strip():
        return "⚠️ 请输入搜索关键词"
    try:
        mgr = RecordManager()
        page_size = 15
        results = []

        if category in ("全部", "简历"):
            r = mgr.resume.search(keyword=keyword, page=page, page_size=page_size)
            for item in r.items:
                results.append({
                    "type": "📄 简历",
                    "id": item.get("id"),
                    "name": item.get("name", "N/A"),
                    "time": item.get("created_at", "N/A"),
                })

        if category in ("全部", "JD"):
            r = mgr.jd.search(keyword=keyword, page=page, page_size=page_size)
            for item in r.items:
                results.append({
                    "type": "📋 JD",
                    "id": item.get("id"),
                    "name": f"{item.get('job_title', 'N/A')} @ {item.get('company', 'N/A')}",
                    "time": item.get("created_at", "N/A"),
                })

        if category in ("全部", "面试"):
            r = mgr.interview.search(job_title=keyword, page=page, page_size=page_size)
            for item in r.items:
                results.append({
                    "type": "🎤 面试",
                    "id": item.get("session_id", "N/A")[:16],
                    "name": item.get("job_title", "N/A"),
                    "time": item.get("started_at", "N/A"),
                })

        if not results:
            return f"未找到与 \"{keyword}\" 相关的结果（第 {page} 页）"

        lines = [f"### 搜索结果: \"{keyword}\" (第 {page} 页)\n",
                 "| 类型 | ID | 名称 | 时间 |",
                 "|------|-----|------|------|"]
        for r_item in results[:page_size]:
            lines.append(f"| {r_item['type']} | {r_item['id']} | {r_item['name']} | {r_item['time']} |")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 搜索失败: {e}"


def build_tab5_history() -> None:
    """构建 Tab5：历史记录"""
    gr.Markdown("## 📜 历史记录")
    gr.Markdown("查看和管理所有历史匹配、优化和面试记录。")

    with gr.Row():
        btn_refresh = gr.Button("🔄 刷新数据", variant="primary")
        btn_export = gr.Button("📤 导出所有数据")
        btn_clear_all = gr.Button("🗑️ 清空全部历史", variant="stop")

    gr.Markdown("---")
    stats_md = gr.Markdown("")

    with gr.Tabs():
        with gr.TabItem("📄 简历记录"):
            resume_md = gr.Markdown("")
        with gr.TabItem("🔧 优化记录"):
            opt_md = gr.Markdown("")
        with gr.TabItem("🎤 面试记录"):
            interview_md = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("### 🔍 搜索")

    with gr.Row():
        search_kw = gr.Textbox(label="搜索关键词", placeholder="输入姓名/职位/公司...", scale=3)
        search_cat = gr.Dropdown(
            choices=["全部", "简历", "JD", "面试"],
            value="全部",
            label="分类",
            scale=1,
        )
        search_page = gr.Number(value=1, label="页码", precision=0, scale=1)
        btn_search = gr.Button("🔍 搜索", variant="secondary", scale=1)

    search_result = gr.Markdown("")

    clear_status = gr.Markdown("")

    # 事件
    btn_refresh.click(
        _history_load_all,
        outputs=[stats_md, resume_md, opt_md, interview_md, clear_status],
    )
    btn_export.click(_history_export, outputs=[clear_status])
    btn_clear_all.click(_history_clear, outputs=[clear_status]).then(
        _history_load_all,
        outputs=[stats_md, resume_md, opt_md, interview_md, clear_status],
    )
    btn_search.click(
        _history_search,
        inputs=[search_kw, search_cat, search_page],
        outputs=[search_result],
    )

    # 初始加载
    initial = _history_load_all()
    stats_md.value = initial[0]
    resume_md.value = initial[1]
    opt_md.value = initial[2]
    interview_md.value = initial[3]


# ============================================================
# Tab6: 自主 Agent（v2.0 - ReAct 自主任务）
# ============================================================

def _format_agentic_trace(trace) -> str:
    """将 agentic_trace（ReActStep 的 dict 列表）格式化为可读轨迹。"""
    if not trace:
        return "_（无推理轨迹）_"
    lines = [f"[ReAct 轨迹] 共 {len(trace)} 步"]
    for s in trace:
        lines.append(f"\nStep {s.get('step_index', '?')}:")
        lines.append(f"  Thought    : {s.get('thought', '')}")
        lines.append(f"  Action     : {s.get('action', '')}")
        action_input = json.dumps(s.get("action_input", {}), ensure_ascii=False)
        lines.append(f"  Action Input: {action_input}")
        obs = str(s.get("observation", "")).replace("\n", "\n    ")
        lines.append(f"  Observation: {obs}")
    return "\n".join(lines)


def _agentic_run(task: str, use_fast_path: bool) -> str:
    """执行自主 Agent 任务，返回结果 + 推理轨迹。"""
    if not task or not task.strip():
        return "⚠️ 请先输入任务描述。"

    try:
        result = run_agentic_task(task=task, fast_path=use_fast_path)
        answer = result.agentic_result
        trace = _format_agentic_trace(result.agentic_trace)
        return f"## 🤖 最终回答\n\n{answer}\n\n---\n\n{trace}"
    except Exception as e:
        logger.error(f"自主 Agent 任务失败: {e}")
        return f"❌ 自主任务执行失败: {e}"


def build_tab6_agentic():
    """自主 Agent Tab：自然语言任务 → ReAct 自主调用工具完成"""
    with gr.Row():
        gr.Markdown(
            """
            ### 🤖 自主 Agent（ReAct）
            > 输入任意自然语言任务，Agent 将自主完成 **思考 → 行动 → 观察** 循环，
            > 调用内置工具（加权评分 / 等级映射 / 匹配报告 / 知识库检索 / 报告导出）完成任务。

            **示例任务**：
            - `技能80 经验60 学历90，计算加权综合得分并给出等级`
            - `生成一份三维度匹配分析报告并导出为 txt 文件`
            - `在知识库中检索 RAG 相关的简历资料`
            """
        )

    with gr.Row():
        agentic_task = gr.Textbox(
            label="任务描述",
            placeholder="例：技能80 经验60 学历90，计算加权综合得分并给出等级",
            lines=3,
            scale=4,
        )
        with gr.Column(scale=1):
            use_fast_path = gr.Checkbox(
                label="快速路径（跳过解析/检索/评分）",
                value=True,
                info="关闭后走完整 LangGraph 图，可复用简历/JD 上下文",
            )
            btn_agentic = gr.Button("🚀 执行自主任务", variant="primary")

    with gr.Row():
        agentic_result = gr.Markdown("")
        agentic_trace = gr.Markdown("")

    btn_agentic.click(
        _agentic_run,
        inputs=[agentic_task, use_fast_path],
        outputs=[agentic_result],
    )


# ============================================================
# 主界面组装
# ============================================================

def create_ui() -> gr.Blocks:
    """创建完整 Gradio 六 Tab 界面"""
    with gr.Blocks(
        title=settings.GRADIO_TITLE,
    ) as demo:
        # Header
        gr.Markdown(
            f"""# 🎯 {settings.GRADIO_TITLE}
            > 基于 LangGraph 多智能体协作的智能求职辅助系统
            ---
            """
        )

        with gr.Tabs():
            with gr.TabItem("📚 知识库管理"):
                build_tab1_knowledge_base()

            with gr.TabItem("📊 匹配分析"):
                build_tab2_match_analysis()

            with gr.TabItem("✨ 简历优化"):
                build_tab3_optimization()

            with gr.TabItem("🎤 模拟面试"):
                build_tab4_interview()

            with gr.TabItem("📜 历史记录"):
                build_tab5_history()

            with gr.TabItem("🤖 自主 Agent"):
                build_tab6_agentic()

    return demo


def launch_ui():
    """启动 Gradio 前端服务"""
    css = """
    .gradio-container { max-width: 1400px !important; }
    footer { display: none !important; }
    """
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald")

    demo = create_ui()
    logger.info(f"Gradio 启动: http://{settings.GRADIO_SERVER_HOST}:{settings.GRADIO_SERVER_PORT}")
    demo.launch(
        server_name=settings.GRADIO_SERVER_HOST,
        server_port=settings.GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    launch_ui()
