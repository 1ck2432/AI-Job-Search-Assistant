# 曹康 - 后端工程师 / AI应用开发工程师

男 / 2005.01
3537216250@qq.com
https://github.com/1ck2432

---

## 教育背景

**湘潭大学 - 人工智能（本科）** 2023.09 - 2027.06
主修课程：程序语言设计（Python、C）、数据结构与算法、计算机网络、MySQL数据库操作、操作系统原理
技能证书：英语四六级

---

## 实践经历

### 武汉中软卓越科技有限公司

**项目描述**：参加人工智能方向企业级实训，系统学习并实践了从物联网底层开发到 AI 大模型应用的全栈技术链路，最终完成"智慧家居系统"综合项目交付。

**工作内容**：
- **AIOT 物联网开发**：基于 STM32 微控制器进行嵌入式 C 语言开发，完成传感器数据采集与串口通信协议调试，实现物联网感知层的硬件交互。
- **PyQt5 桌面应用开发**：使用 PyQt5 框架开发跨平台 GUI 应用，掌握信号槽机制、多窗口切换与 UI 设计规范。
- **智能语音处理**：基于科大讯飞 SDK 实现语音唤醒、语音识别（ASR）与语音合成（TTS），完成语音交互模块开发。
- **AI 大模型应用**：使用 Ollama 完成大模型本地离线部署；基于 LoRA 进行模型轻量级微调；使用 Dify 平台与 LangChain 框架开发 RAG 知识库问答应用。

---

## 项目

### ResumeAgent - AI 多智能体求职助手

Python LangGraph LangChain Gradio ChromaDB SQLite

**项目描述**：
独立开发并部署的基于 LangGraph 多智能体架构的求职辅助系统，编排简历解析、岗位匹配、简历优化、AI 模拟面试、自主 Agent 等多个专业化 Agent 协作，形成覆盖求职全流程的一站式平台。

**工作内容**：
- **多智能体编排**：基于 LangGraph 状态图构建解析、检索、评分、优化、面试、摘要、自主 Agent 七大节点工作流，评分后三向条件路由分流，实现任务状态流转与多轮迭代优化；
- **Function Calling 结构化输出**：LLM 层封装 ToolDefinition / ToolCall 统一抽象，通过 LangChain bind_tools 接入多 Provider；自研 pydantic_model_to_tool 将 Pydantic Schema 强制转为 tool_call（tool_choice 锁定函数名），替代脆弱的 JSON 文本解析，评分 / 面试出题 / 回答评估三节点复用同一封装，失败自动回退 JSON 解析；
- **自研 ReAct 引擎**：实现 Thought→Action→Observation 循环，内置工具注册表（加权评分 / 等级映射 / 匹配报告 / RAG 检索 / 报告导出），支持参数别名归一化容错模型乱改键名、连续重复动作检测与步数上限，经 agentic 节点接入 LangGraph 构成第 7 节点，UI 输入自然语言任务即可自主完成闭环；
- **RAG 检索增强**：基于 ChromaDB 构建向量知识库，集成 BGE-M3 文本嵌入与混合检索策略，实现简历与 JD 的语义匹配与相关片段召回；
- **简历智能优化**：设计关键词匹配 / 量化成果 / 精简表达三种优化模式，支持多轮迭代优化与优化前后高亮对比，一键导出 Word 报告；
- **AI 模拟面试**：实现流式对话交互、逐轮自动评分与面试复盘报告自动生成；
- **前端与工程化**：使用 Gradio 搭建六 Tab 交互界面（含自主 Agent 任务与轨迹展示），基于 SQLite（WAL 模式）持久化历史记录，支持 Ollama / OpenAI / DeepSeek 多 LLM 无缝切换，LLM 调用自动重试 + 指数退避保障服务稳定。

### SimpleSocial 全栈图文视频社交平台

Python FastAPI Streamlit MySQL SQLAlchemy

**项目描述**：
独立开发并部署的前后端分离社交网络平台。系统支持用户安全注册登录、多媒体动态（图片/视频）发布与删除，并实现了基于第三方云端的媒体实时渲染与分发。

**工作内容**：
- 基于 FastAPI 搭建异步后端接口，运用 JWT 实现身份认证、权限拦截与密码哈希校验；
- 设计 MySQL 数据表，通过 SQLAlchemy + aiomysql 完成异步 ORM 建模与数据增删改查；
- 借助 Streamlit 搭建前端页面，完成文件传输与状态管理，实现全业务流程。

---

## 技能

- **编程语言**：精通 Python，掌握 C 语言，熟练使用异步编程、面向对象开发
- **后端开发**：熟悉 FastAPI、Uvicorn，了解 RESTful API 设计与接口联调
- **数据库**：熟练使用 MySQL，掌握 SQLAlchemy 异步 ORM 开发，了解 SQLite
- **AI 应用**：掌握 LangChain、LangGraph 多智能体编排、RAG 检索增强、ChromaDB 向量库、Gradio 应用开发
- **Agent 开发**：掌握 Function Calling / Tool Calling 与结构化输出、ReAct 推理引擎实现、工具注册与分发、多 Agent 协作
- **大模型应用**：熟悉 Ollama 本地部署、LoRA 微调、多 LLM 接入（通义千问 / DeepSeek / OpenAI）
