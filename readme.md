好的，这是为 `backend` 目录编写的 `README.md` 文件。

---

# Nexus Backend

这个目录包含了 Nexus Copilot 应用的 FastAPI 后端服务。它负责处理所有复杂的、长时间运行的和数据密集型的操作，为 Tauri 前端提供一个健壮的 API。

## 核心功能

-   **🤖 Agentic 框架 (Agentic Framework):** 使用不同的模式（Plan, Explore, Write, Research, Debate）来编排和执行多步骤的 AI 代理任务。
-   **🔄 LLM & API 代理 (LLM & API Proxy):** 提供一个统一的接口，用于与各种外部 LLM 和创意 AI API 进行通信（用于聊天、嵌入、图像/音频/视频生成），并在此过程中注入 RAG 上下文。
-   **🧠 向量数据库服务 (Vector Database Service):** 管理一个持久化的 [ChromaDB](https://www.trychroma.com/) 实例，用于高效的相似性搜索和检索增强生成 (RAG)。
-   **📚 知识库管理 (Knowledge Base Management):** 处理文档解析（PDF, DOCX 等）、文本分块、嵌入向量生成和索引。
-   **🕸️ 知识图谱 (Knowledge Graph):** 解析 Markdown 笔记中的 `[[WikiLinks]]`，以构建和维护一个由相互关联的知识组成的网络图。
-   **📊 数据与系统服务 (Data & System Services):** 提供用于应用仪表盘统计、全量数据备份/恢复以及批量文档转换的 API 端点。
-   **🎨 创意内容生成 (Creative Content Generation):** 代理对外部服务的请求，以生成图像、音频和视频等创意内容。

## 技术栈

-   **Web 框架:** [FastAPI](https://fastapi.tiangolo.com/)
-   **ASGI 服务器:** [Uvicorn](https://www.uvicorn.org/)
-   **向量存储:** [ChromaDB](https://www.trychroma.com/)
-   **结构化数据存储:** [SQLite](https://www.sqlite.org/index.html)
-   **文本处理:** [LangChain](https://www.langchain.com/) (用于文本分割)
-   **数据验证:** [Pydantic](https://docs.pydantic.dev/)

## 项目结构

```
backend/
├── app/
│   ├── api/              # FastAPI 路由和端点定义
│   ├── agents/           # Agentic 框架的核心逻辑，包括模式和提示
│   ├── core/             # 应用配置 (如数据路径)
│   ├── database/         # SQLite 数据库连接和初始化
│   ├── knowledge_base/   # 文档索引和搜索相关逻辑
│   ├── schemas/          # Pydantic 模型，用于 API 请求/响应验证
│   └── services/         # 核心业务逻辑 (与数据库、外部 API 交互等)
├── files/                # (运行时创建) 存储 Agent 生成的日志和报告
├── tasks/                # (运行时创建) 存储 Agent 生成的可预览文件 (如 HTML)
├── chroma_data/          # (运行时创建) ChromaDB 的持久化数据
├── nexus.sqlite          # (运行时创建) SQLite 数据库文件
└── main.py               # FastAPI 应用主入口
```

## 安装与运行

该后端服务被设计为由主 Nexus Tauri 应用程序自动管理（安装、启动和停止）。但为了独立开发和测试，您可以按照以下步骤手动运行它：

1.  **先决条件:**
    -   Python 3.10 或更高版本
    -   Git

2.  **克隆依赖 (如果尚未安装):**
    后端依赖于一个 Git 子模块来获取其核心依赖。请确保它已被克隆：
    ```bash
    git submodule update --init --recursive
    ```

3.  **创建虚拟环境:**
    ```bash
    python -m venv venv
    ```

4.  **激活虚拟环境:**
    -   在 macOS/Linux 上: `source venv/bin/activate`
    -   在 Windows 上: `venv\Scripts\activate`

5.  **安装依赖:**
    后端需要从 `requirements.txt` 文件安装依赖项。
    ```bash
    pip install -r requirements.txt
    ```

6.  **运行开发服务器:**
    ```bash
    uvicorn app.main:app --reload --port 8008
    ```
    服务器现在应该在 `http://127.0.0.1:8008` 上运行。

### 配置说明

后端通过 `ApiConfig` schema 从 Tauri 前端接收所有必要的配置（如 API 密钥）。它被设计为无状态的，不直接存储敏感密钥。

`NEXUS_DATA_PATH` 环境变量由 Tauri 应用在启动时设置。在独立开发模式下，此路径默认为 `backend/` 目录本身。所有持久化数据（SQLite 数据库、ChromaDB 数据、Agent 文件）都将存储在此目录中。

## API 概览

所有端点都位于 `/api/v1` 前缀下。

-   `/agent`: 启动、停止、恢复和获取 Agent 任务的状态。
-   `/proxy`: 安全地代理对外部 LLM API 的请求，处理 RAG 上下文注入和身份验证。
-   `/vector`: 与 ChromaDB 向量存储进行交互（添加、查询、删除向量）。
-   `/knowledge_base`: 处理文件以进行索引，并管理知识图谱链接。
-   `/creations`: 生成创意内容，如图像和音频。
-   `/dashboard`: 为仪表盘 UI 提供统计数据。
-   `/backup`: 处理应用程序数据的完整导出和导入。
-   `/convert`: 提供用于批量文档转换的流式端点。

## 工作流示例：一个 Agent 任务的生命周期

1.  Tauri 前端向 `/api/v1/agent/start-task` 发送一个请求，其中包含用户的目标和当前的 `ApiConfig`。
2.  后端在 SQLite 中创建一个新的任务记录，并使用 `BackgroundTasks` 启动一个后台作业。
3.  `agents.runner` 接管任务，并根据请求的 `mode`（例如 `write_mode`）委托给相应的执行器。
4.  Agent 模式通过调用 `/proxy` 端点来编排对 LLM 的调用，以生成计划、策略和内容。
5.  中间结果和最终报告被保存到 SQLite 数据库和文件系统（在 `files/` 目录下）。
6.  Tauri 前端轮询 `/api/v1/agent/get-task-status/{task_id}` 端点以获取实时更新，并将其显示给用户。