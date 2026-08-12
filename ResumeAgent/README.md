# ResumeAgent - AI 多智能体求职助手

> 基于 LangGraph 多智能体协作的智能求职辅助系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)](https://www.gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)](https://www.trychroma.com/)

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [运行示例](#运行示例)
- [模块说明](#模块说明)
- [集成测试](#集成测试)
- [常见问题](#常见问题)

---

## 项目简介

ResumeAgent 是一个基于多智能体（Multi-Agent）架构的求职辅助平台，利用 LangGraph 编排多个专业化 Agent 协作完成简历解析、岗位匹配、简历优化、AI 模拟面试等全流程求职辅助任务。

### 技术栈

| 组件 | 技术选型 |
|------|----------|
| 前端 | Gradio 6.x（六 Tab 交互界面） |
| 智能体编排 | LangGraph（状态图工作流，7 节点 + 条件路由） |
| Agent 引擎 | 自研 ReActAgent（Thought/Action/Observation 循环） |
| LLM 后端 | Ollama / OpenAI / DeepSeek 多模式切换（Function Calling） |
| 向量检索 | ChromaDB + Sentence-Transformers |
| 数据持久化 | SQLite（WAL 模式） |
| 图表渲染 | Plotly（雷达图/柱状图） |
| 日志系统 | Loguru（分级文件 + 控制台） |

---

## 功能特性

### 六大核心模块

1. **知识库管理** - 上传文件入库、向量检索测试、库统计可视化
2. **简历 & JD 匹配分析** - 智能解析、多维评分、雷达图/柱状图、缺失技能表
3. **简历智能优化** - 高亮对比、三种优化模式、一键导出 Word
4. **AI 模拟面试** - 流式对话、逐轮评分、复盘报告下载
5. **历史记录** - SQLite 数据查询展示、统计概览
6. **自主 Agent（v2.0）** - 输入自然语言任务，自研 ReActAgent 自主调度工具完成并展示推理轨迹

### 特色能力

- **多 LLM 兼容**: 本地 Ollama + 云端 OpenAI/DeepSeek 无缝切换
- **Function Calling**: ToolDefinition/ToolCall 统一抽象，pydantic_model_to_tool 强制结构化输出
- **自研 ReAct 引擎**: Thought→Action→Observation 循环、工具注册表、重复动作检测、参数别名归一化
- **检索引擎**: RAG 技术结合向量搜索，自动召回相关知识库片段
- **分数量化**: 技能/经验/学历三维加权评分，自动生成雷达图
- **流式输出**: 面试对话支持 SSE 流式响应对接
- **异常恢复**: LLM 调用自动重试 + 指数退避，确保服务稳定

---

## 系统架构

```
用户 (Gradio Web UI)
│
├─ Tab1: 知识库管理 ────────────┐
├─ Tab2: 简历&JD匹配分析 ───────┤
├─ Tab3: 简历智能优化 ──────────┤
├─ Tab4: AI模拟面试 ────────────┤
├─ Tab5: 历史记录 ──────────────┤
└─ Tab6: 自主Agent任务 ─────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│           LangGraph Workflow             │
│                                          │
│  parse_node → retrieve_node → score_node │
│       │              │              │    │
│       ▼              ▼              ▼    │
│  TextSplitter   HybridRetriever   Rule+  │
│  DocumentLoader  ChromaDB        Weighted│
│                                          │
│  optimize_node → interview_generate      │
│       │              │                   │
│       ▼              ▼                   │
│  三种模式优化    逐轮面试+评分           │
│                                          │
│  summary_node (复盘报告生成)             │
│                                          │
│  agentic_node (v2.0 自研 ReActAgent)     │
│  ─ 读取 agentic_task，Thought/Action/    │
│    Observation 循环，调度内置工具自主完成 │
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│              基础设施层                    │
│  ┌──────┐ ┌────────┐ ┌────────────────┐  │
│  │ LLM  │ │向量数据库│ │  SQLite 数据库  │  │
│  │工厂  │ │ChromaDB│ │  (WAL模式)     │  │
│  └──────┘ └────────┘ └────────────────┘  │
│  ┌──────┐ ┌────────┐ ┌────────────────┐  │
│  │文件  │ │评分工具 │ │  日志系统       │  │
│  │解析  │ │+图表   │ │  Loguru        │  │
│  └──────┘ └────────┘ └────────────────┘  │
└──────────────────────────────────────────┘
```

### 目录结构

```
ResumeAgent/
├── main.py                   # 系统入口（健康检查、日志初始化、UI启动）
├── test_integration.py       # 全链路集成测试脚本
├── requirements.txt          # Python 依赖清单
├── .env.example              # 环境变量模板（复制为 .env 后填写）
├── .gitignore                # Git 忽略规则（.env/运行数据不提交）
├── config/
│   └── settings.py           # 全局配置（Pydantic Settings，读取根目录 .env）
├── core/
│   ├── llm/
│   │   ├── base.py           # LLM 抽象基类 + 重试机制 + Function Calling (v2.0)
│   │   ├── ollama_llm.py     # Ollama 本地推理（bind_tools）
│   │   ├── openai_llm.py     # OpenAI / DeepSeek API（bind_tools）
│   │   └── __init__.py       # LLM 工厂
│   ├── agents/               # v2.0 自研 ReAct 引擎
│   │   ├── react_agent.py    # ReActAgent：Thought/Action/Observation 循环
│   │   ├── tool_registry.py  # 工具注册表：ReActTool + builtin_tools
│   │   └── __init__.py
│   ├── graph/
│   │   ├── agent_state.py    # AgentState Pydantic 数据模型
│   │   ├── agent_nodes.py    # LangGraph 节点实现（Function Calling 结构化输出）
│   │   ├── agentic_node.py   # 第 7 节点：自主 Agent (v2.0)
│   │   └── workflow_graph.py # 状态图构建 + run_agentic_task 封装
│   ├── rag/
│   │   ├── document_loader.py # 文档加载与清洗
│   │   ├── text_splitter.py  # 语义切片
│   │   ├── vector_store.py   # ChromaDB 向量库
│   │   └── retriever.py      # RAG 检索器
│   ├── tools/
│   │   ├── sqlite_db.py      # 移除（使用 database/db.py）
│   │   ├── file_export.py    # Word/TXT 导出
│   │   └── score.py          # 评分计算工具
│   └── utils/
│       ├── logger.py          # 日志工具（装饰器、检查）
│       └── file_parser.py     # 多格式文件解析
├── database/
│   └── db.py                 # SQLite ORM 管理层
├── webui/
│   └── gradio_ui.py          # Gradio 六 Tab 界面（含自主 Agent Tab）
├── examples/
│   └── react_demo.py         # v2.0 ReAct 命令行演示
├── uploads/                   # 上传文件目录
├── chroma_db/                 # 向量库持久化目录
├── logs/                      # 日志文件目录
└── exports/                   # 导出文件目录
```

---

## 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/)（本地 LLM 模式）或 OpenAI API Key（云端模式）
- 8GB+ 内存（推荐，用于本地 Embedding 模型）

### 1. 克隆项目 & 进入目录

```bash
cd "AI 多智能体求职助手/ResumeAgent"
```

### 2. 创建虚拟环境 & 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env` 模板并编辑：

```bash
# 复制配置模板（项目根目录下执行）
copy .env.example .env   # Windows
# cp .env.example .env    # macOS/Linux
```

`.env` 文件关键配置项：

```env
# ==================== LLM 模式配置 ====================
# 三选一: ollama / openai / deepseek
LLM_PROVIDER=ollama

# -- Ollama 本地 --
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b

# -- OpenAI API --
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# -- DeepSeek API --
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat

# ==================== 其他配置 ====================
LOG_LEVEL=INFO
GRADIO_SERVER_HOST=127.0.0.1
GRADIO_SERVER_PORT=7860
```

### 4. 启动服务

```bash
python main.py
```

启动成功后会输出：

```
╔══════════════════════════════════════════════════════════╗
║  🎯 ResumeAgent - AI 多智能体求职助手                   ║
╚══════════════════════════════════════════════════════════╝

17:30:01 | INFO     | 日志系统初始化完成
17:30:01 | INFO     | SQLite 数据库初始化完成
17:30:02 | INFO     | ChromaDB向量库: 已连接 (0 条记录)
17:30:02 | INFO     | Gradio 界面构建完成，正在启动服务...
```

浏览器访问 **http://127.0.0.1:7860** 即可使用。

---

## 配置说明

### 完整配置项

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `ollama` | LLM 提供商标识: ollama/openai/deepseek |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama 模型名 |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 模型 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 代理地址 |
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型 |
| `LOG_LEVEL` | `INFO` | 日志级别: DEBUG/INFO/WARNING/ERROR |
| `LOG_RETENTION` | `7 days` | 日志保留天数 |
| `GRADIO_SERVER_HOST` | `127.0.0.1` | Web 服务绑定地址 |
| `GRADIO_SERVER_PORT` | `7860` | Web 服务端口 |
| `RAG_CHUNK_SIZE` | `500` | 文档切片大小 |
| `RAG_CHUNK_OVERLAP` | `50` | 切片重叠大小 |
| `EMBED_MODEL_NAME` | `shibing624/text2vec-base-chinese` | Embedding 模型 |
| `EMBED_DEVICE` | `cpu` | Embedding 设备: cpu/cuda |

### LLM 模式切换

**本地 Ollama 模式**（推荐开发/隐私场景）：

```bash
# 1. 安装并启动 Ollama
ollama serve

# 2. 拉取模型（推荐中文模型）
ollama pull qwen3:8b

# 3. 配置 .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
```

**OpenAI 云端模式**：

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
```

**DeepSeek 模式**（高性价比）：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
```

---

## 运行示例

### 启动系统

```bash
cd ResumeAgent
python main.py
```

### 运行集成测试

```bash
# 完整测试（包含 LLM 连通性检查）
python test_integration.py

# 跳过 LLM 联网测试
SKIP_LLM_TESTS=1 python test_integration.py

# 跳过向量库测试
SKIP_VS_TESTS=1 python test_integration.py

# 开启详细日志
VERBOSE=1 python test_integration.py

# 首失败即终止
FAIL_FAST=1 python test_integration.py
```

### 各 Tab 使用示例

#### Tab1: 知识库管理

1. 上传 PDF/DOCX/TXT 格式的行业知识文档
2. 点击「入库」存入向量数据库
3. 在检索框中输入查询语句测试检索效果
4. 查看知识库统计图表

#### Tab2: 简历 & JD 匹配分析

1. 上传简历文件（PDF/Word/TXT）或粘贴文本
2. 上传 JD 文件或粘贴文本
3. 点击「开始分析」
4. 查看匹配总分、三维雷达图、技能匹配柱状图、缺失技能列表

#### Tab3: 简历智能优化

1. 在 Tab2 完成匹配分析后自动带入
2. 或手动上传简历 + JD
3. 选择优化模式：
   - **针对性优化**: 聚焦 JD 要求
   - **全面优化**: 兼顾表达能力
   - **简洁优化**: 精简篇幅
4. 点击「一键优化」，左侧原稿 + 右侧优化稿对比
5. 可反复迭代优化
6. 点击「下载 Word」导出

#### Tab4: AI 模拟面试

1. 填入 JD 内容（可选附带简历）
2. 设置面试题数（3-10 题）
3. 点击「开始面试」
4. 在聊天框输入回答，AI 评分并给出下一题
5. 可随时暂停/继续
6. 点击「结束面试」生成复盘报告并下载

#### Tab5: 历史记录

1. 查看历史简历、JD、优化记录、面试记录
2. 使用筛选条件查询
3. 查看系统使用统计

---

## 模块说明

### AgentState（状态模型）

```python
from core.graph.agent_state import AgentState

state = AgentState(
    resume_raw="...",     # 原始简历文本
    jd_raw="...",         # 原始 JD 文本
)
# 通过 model_copy() 更新状态
state = state.model_copy(update={"match_score": MatchScoreDetail(...)})
```

### LLM 工厂

```python
from core.llm import get_llm

llm = get_llm()  # 自动根据 settings.LLM_PROVIDER 选择
resp = llm.chat_with_prompt("你好", with_retry=True)
```

### 日志装饰器

```python
from core.utils.logger import safe_execute, log_execution_time, TimingContext

@safe_execute(default_return="操作失败")
def risky_operation(x):
    ...

with TimingContext("LLM 推理"):
    result = llm.chat(...)
```

### 数据库操作

```python
from database.db import get_db

db = get_db()
stats = db.get_statistics()
# {'resume_count': 5, 'jd_count': 3, ...}
```

---

## 集成测试

测试覆盖 13 项检查，确保核心链路可用：

```bash
$ python test_integration.py

============================================================
  ResumeAgent 全链路集成测试
============================================================

✅  模块导入: 12 个核心模块导入成功
✅  配置加载: LLM=ollama  Level=INFO
✅  日志工具: 装饰器/上下文管理器正常
✅  文档清洗: 清洗后 234 字符
✅  文档切片: 3 个切片
✅  Pydantic 状态模型: 创建/更新/属性访问正常
✅  评分计算/等级映射: 加权=79.8 等级=较高匹配
✅  LLM 异常体系: 异常体系完整
✅  SQLite CRUD: resume=1 jd=1 session=abc123
✅  ChromaDB 向量库: 向量库记录数: 0
✅  文件解析(TXT): 234 字符
✅  简历导出: TXT 导出正常
✅  LLM 工厂: provider=ollama model=qwen3:8b

============================================================
  测试报告: 13/13 通过, 0 失败
============================================================

🎉 所有测试通过！系统各模块工作正常。
```

---

## 常见问题

### 1. Ollama 连接失败

```
错误: HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded
```

**解决**: 确保 Ollama 服务正在运行

```bash
# 启动 Ollama 服务
ollama serve

# 验证服务正常
curl http://127.0.0.1:11434/api/tags
```

### 2. Embedding 模型下载慢

首次启动会自动下载中文 Embedding 模型（约 400MB），请耐心等待。也可手动预下载：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"
```

### 3. Gradio 端口被占用

```env
# 在 .env 中修改端口
GRADIO_SERVER_PORT=7861
```

### 4. 词向量维度不匹配

如遇 `chromadb.errors.InvalidDimensionException`，清除旧向量库：

```bash
# 删除旧向量库文件
rm -rf ResumeAgent/chroma_db
```

### 5. Word 文档导出报错

需要安装 `python-docx`:

```bash
pip install python-docx
```

### 6. 查看完整日志

```bash
# 查看当天日志
cat ResumeAgent/logs/resume_agent_$(date +%Y-%m-%d).log

# 查看错误日志
cat ResumeAgent/logs/error_$(date +%Y-%m-%d).log
```

---

## 开发计划

- [ ] Docker 一键部署
- [ ] 多用户 Web 鉴权
- [ ] 面试题库可配置化
- [ ] 简历模板市场
- [ ] 定时推送匹配岗位

---

## License

MIT License
