# -*- coding: utf-8 -*-
"""
将"优秀简历.jpg"中的内容转为结构化 PDF。
版式按原图：实习经历 / 项目经历 / 荣誉技能 三段式。
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------- 中文字体 ----------
FONT_DIR = r"C:\Windows\Fonts"
# 优先使用思源/Noto/Microsoft YaHei，依次回退
for cand in [
    ("MicrosoftYaHei",      FONT_DIR + r"\msyh.ttc"),
    ("MicrosoftYaHeiUI",    FONT_DIR + r"\msyhui.ttc"),
    ("SimSun",              FONT_DIR + r"\simsun.ttc"),
    ("SimHei",              FONT_DIR + r"\simhei.ttf"),
    ("NotoSansCJK",         FONT_DIR + r"\NotoSansCJK-Regular.ttc"),
    ("NotoSerifCJK",        FONT_DIR + r"\NotoSerifCJK-Regular.ttc"),
]:
    name, path = cand
    try:
        pdfmetrics.registerFont(TTFont(name, path))
        ZH_FONT = name
        break
    except Exception:
        continue
else:
    raise RuntimeError("未找到可用的中文字体，请安装微软雅黑或 Noto CJK。")

# ---------- 样式 ----------
title_style = ParagraphStyle(
    "Title", fontName=ZH_FONT, fontSize=15, leading=20,
    textColor=colors.HexColor("#1F4E79"), spaceBefore=4, spaceAfter=6,
)
company_style = ParagraphStyle(
    "Company", fontName=ZH_FONT, fontSize=11, leading=15,
    textColor=colors.HexColor("#1F4E79"), spaceBefore=4, spaceAfter=2,
)
date_style = ParagraphStyle(
    "Date", fontName=ZH_FONT, fontSize=10, leading=14,
    textColor=colors.HexColor("#555555"), alignment=TA_LEFT,
)
tech_style = ParagraphStyle(
    "Tech", fontName=ZH_FONT, fontSize=10, leading=14,
    textColor=colors.HexColor("#333333"), spaceBefore=1, spaceAfter=2,
)
bullet_style = ParagraphStyle(
    "Bullet", fontName=ZH_FONT, fontSize=10, leading=14,
    leftIndent=12, bulletIndent=0, spaceBefore=1, spaceAfter=2,
)
project_desc_style = ParagraphStyle(
    "ProjDesc", fontName=ZH_FONT, fontSize=10, leading=14,
    textColor=colors.HexColor("#333333"), spaceBefore=2, spaceAfter=4,
)
section_rule = ParagraphStyle(
    "Rule", fontName=ZH_FONT, fontSize=1, leading=1, spaceBefore=2, spaceAfter=4,
)

styles = getSampleStyleSheet()
normal = ParagraphStyle("Body", parent=styles["BodyText"],
                       fontName=ZH_FONT, fontSize=10, leading=14)

# ---------- 内容数据 ----------
content = []

# ============== 实习经历 ==============
content.append(Paragraph("实习经历", title_style))
content.append(Paragraph('<font color="#999999">―</font>' * 90, section_rule))

# 实习 1：原图公司名模糊，使用占位
content.append(Paragraph("（公司名）全栈开发 &nbsp;&nbsp;&nbsp; <font color='#888888'>2026.05 ~ 2026.08</font>", company_style))
content.append(Paragraph(
    "<b>技术栈：</b>React 17、Ant Design Mobile、微信 JS-SDK、Spring Boot、MyBatis-Plus、MySQL、Redis、OpenFeign、Eureka、FastDFS",
    tech_style,
))
content.append(Paragraph(
    "• 负责 ERP 系统的服务端工单模块全栈开发，使用 React、Spring Boot、MyBatis-Plus 实现覆盖 3 类角色、10+ 状态节点的工单闭环，结合角色权限校验、幂等操作，保障不同业务节点的数据一致性与操作安全。",
    bullet_style,
))
content.append(Paragraph(
    "• 优化服务端工单数据模型，拆分工单级结果与链路级故障明细，完善事务控制及数据校验，减少约 30% 的重复录入。",
    bullet_style,
))
content.append(Spacer(1, 6))

# 实习 2：恒生电子
content.append(Paragraph("恒生电子股份有限公司&nbsp;&nbsp; JAVA 后端开发 &nbsp;&nbsp;&nbsp; <font color='#888888'>2025.03 ~ 2025.07</font>", company_style))
content.append(Paragraph(
    "<b>技术栈：</b>Spring Boot2、MyBatis-Plus、Spring Security、JWT、MySQL、Redis、Nginx、OSS、Postman",
    tech_style,
))
content.append(Paragraph(
    "• 参与企业业务系统后端研发（因子开发），负责业务接口、数据库操作及技术文档编写，协助多部门完成系统测试。",
    bullet_style,
))
content.append(Paragraph(
    "• 基于 Spring Boot、MyBatis-Plus、MySQL 开发业务模块，并通过 JWT、HttpOnly Cookie、Spring Security、AOP 等技术实现身份认证与权限管理。",
    bullet_style,
))
content.append(Paragraph(
    "• 对接阿里云 OSS 实现影像文件上传，使用 Nginx 完成反向代理与负载均衡。",
    bullet_style,
))
content.append(Spacer(1, 10))

# ============== 项目经历 ==============
content.append(Paragraph("项目经历", title_style))
content.append(Paragraph('<font color="#999999">―</font>' * 90, section_rule))

# 项目 1：多角色 Agent 协助助手
content.append(Paragraph("多角色 Agent 协助助手 &nbsp;&nbsp;&nbsp; <font color='#888888'>2026.02 ~ 2026.07</font>", company_style))
content.append(Paragraph(
    "<b>项目描述：</b>采用 OpenSpec 技术，面向多角色 NPC 场景，开发支持单角色、多角色编排、长期记忆、个人知识库及 MCP 工具调用的多用户 Agent 平台。系统支持角色人设与权限配置，通过混合检索和分层记忆维持长期对话一致性，并对索引、外部工具和模型异常提供降级及恢复机制。",
    project_desc_style,
))
content.append(Paragraph(
    "<b>技术栈：</b>Python、FastAPI、Vue3、TypeScript、Pinia、PostgreSQL、SQLAlchemy Async、Qdrant、Neo4j、Docker、OpenAI-compatible API",
    tech_style,
))
content.append(Paragraph(
    "• <b>优化 RAG 检索链路：</b>针对 PDF 表格解析丢失、跨段证据不完整及 Dense 检索排序靠后的问题，设计 pypdf→pdflumber 逐页跨级解析、相邻上下文向量化、查询扩展和字段加权重排，构建 \"Top-100 召回、去重重排后 Top-6 输出\" 的两阶段 RAG 链路；在黄金证据开发集上将 Hit@6/Recall@6 由 40% 提升至 100%，平均证据覆盖率由 50.2% 提升至 96.7%。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>融合向量与图谱检索：</b>针对固定长度切分造成语义断裂、单向召回难以整体关联的问题，设计段落与句块边界切分、重叠窗口及相邻上下文扩展，并融合 Qdrant 向量检索与 Neo4j 图谱证据，通过查询意图路由、词项覆盖重排和结果去重提升证据相关性。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>实现多角色编排与恢复：</b>针对多角色对话中发言混乱、重复生成和失败后无法恢复的问题，实现手@、@提及、轮询和 LLM 自动路由，并将发言结果持久化为 SpeakerPlan，通过诺言幂等与执行进度管理保证编排过程可控、可追踪、可恢复。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>构建分层记忆隔离机制：</b>针对长对话上下文持续增长及角色记忆串扰问题，采用 \"近期消息、滚动摘要、结构化长期记忆\" 的分层上下文策略，并通过会话共享与角色私有作用域隔离记忆，在控制上下文长度同时保持人设和历史事实一致。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>设计全链路降级容错：</b>针对模型、向量库、图数据库和 MCP 工具存在独立故障的问题，以 PostgreSQL 作为事实数据源，通过后台任务、失败重试、功能开关和可达能力降级，使图谱或工具不可用时仍能保留基础文本检索和对话能力。",
    bullet_style,
))
content.append(Spacer(1, 8))

# 项目 2：个人医疗助手
content.append(Paragraph("个人医疗助手 &nbsp;&nbsp;&nbsp; <font color='#888888'>2025.09 ~ 2026.01</font>", company_style))
content.append(Paragraph(
    "<b>项目描述：</b>基于大语言模型的智能医疗服务平台，利用大模型实现健康咨询、智能分诊、挂号预约等一站式就医辅助功能。",
    project_desc_style,
))
content.append(Paragraph(
    "<b>技术栈：</b>Spring Boot3、Spring WebFlux、LangChain4j、Vue3、MySQL、MongoDB、Pinecone、Ollama、MyBatis-Plus",
    tech_style,
))
content.append(Paragraph(
    "• <b>基于 RAG 的智能分诊系统：</b>将医院科室文档向量化存入 Pinecone，用户描述症状后自动检索 + 大模型推理，分诊准确率从纯模型基线的 76% 提升至 90%，1.4 个科室覆盖，平均检索延迟 42ms。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>AI Agent 工具调用闭环：</b>基于 LangChain4j 框架实现 Function Calling，定义挂号、查询、取消三类 Tool。AI 自主判别用户意图并调用对应工具，覆盖预约全流程 5 个步骤，告警硬编码 if-else。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>自研记忆存储中间件：</b>实现 LangChain4j ChatMemoryStore 接口，将对话记忆从 Default Memory 迁移至 MongoDB，解决服务重启后会话丢失问题。单用户支持 20+ 轮多轮对话，查询响应 &lt; 5ms。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>流式对话 + 会话管理：</b>WebFlux 实现 SSE 流式响应，首字延迟 &lt; 600ms；侧边栏管理历史会话，单用户 120+ 会话持久化存储。",
    bullet_style,
))
content.append(Spacer(1, 10))

# ============== 荣誉技能 ==============
content.append(Paragraph("荣誉技能", title_style))
content.append(Paragraph('<font color="#999999">―</font>' * 90, section_rule))

content.append(Paragraph(
    "• <b>专业证书：</b>CET-4、中级软件设计师。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>专业技能：</b>编程语言（C/C++、Java、Python），前端（HTML、CSS、JavaScript），后端（Spring Boot、FastAPI），数据库（MySQL），中间件（Redis），其他（Claude Code、Postman、Git、Docker、Linux）。",
    bullet_style,
))
content.append(Paragraph(
    "• <b>在校荣誉：</b>中国研究生数学建模竞赛全国二等奖（2025）、华中科技大学数学建模挑战赛全国三等奖（2025），中国高校计算机大赛·天梯赛总决赛全国三等奖（2023），第二届程序设计大赛一等奖（2022），校奖学金一等奖等。",
    bullet_style,
))

# ---------- 生成 PDF ----------
OUT = r"C:\Users\35372\Desktop\AI 多智能体求职助手\优秀简历.pdf"
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=1.6*cm, bottomMargin=1.6*cm,
    title="优秀简历",
    author="ResumeAgent",
)
doc.build(content)
print(f"OK: {OUT}")