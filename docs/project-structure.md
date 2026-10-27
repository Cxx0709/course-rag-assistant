# 配置与 Web 模块组织

本轮优先处理配置分散、Web 职责混合和静态资源布局。核心 RAG 模块、问题集、报告和脚本的现有路径保留；历史报告仍描述当时的代码版本。

## 当前模块边界

```text
src/course_rag_assistant/
├── config.py              # 类型化配置、环境变量读取与校验
├── answer.py              # 生成、选句、引用验证
├── prerequisites.py       # 显式先修关系、依赖遍历与原文定位
├── compare.py             # 同概念原文筛选、阅读维度与文字差异
├── dense.py / rerank.py   # 现有外部检索服务客户端
├── cli.py                 # 命令行入口
└── web/
    ├── __init__.py        # 保留 create_app、main、Document、Library 导入
    ├── __main__.py        # python -m course_rag_assistant.web
    ├── app.py             # 应用组装与本机服务器入口
    ├── constants.py       # 资源路径和上传策略
    ├── library.py         # 资料保存、校验、重启恢复和检索缓存
    ├── schemas.py         # HTTP 请求模型
    ├── routes.py          # 状态、资料和问答接口
    ├── answers.py         # 网页问答执行与引用响应展示
    ├── prerequisites.py   # 所选资料范围内的复习路径与链接展示
    ├── compare_routes.py  # 两份资料独立检索与对比接口
    ├── quiz_routes.py     # 练习、判分和错题复习接口
    ├── security.py        # 来源检查、令牌和安全响应头
    ├── static/
    │   ├── css/style.css
    │   ├── js/app.js
    │   └── sample/example.md
    └── templates/index.html
```

首页仍由 `FileResponse` 返回静态 HTML，没有新增模板引擎。网页加载 `/assets/css/style.css` 和 `/assets/js/app.js`，旧的 `/assets/style.css`、`/assets/app.js` 继续可用。打包资源清单在 `pyproject.toml` 中显式列出。

启动方式不变：`start-web.cmd`、`start-web.ps1`、`course-rag-web` 和 `python -m course_rag_assistant.web`。网页 API、上传资料目录及 Quiz SQLite 格式保持原样。

## 统一配置

使用标准库的不可变 `Settings`，没有新增运行时依赖。所有 Python 应用代码对 `COURSE_RAG_*` 环境变量的读取集中到 `config.py`。工厂函数仍叫 `generator_from_env`、`embedder_from_env`、`reranker_from_env`，也可以直接注入 `Settings`。

```python
from pathlib import Path
from course_rag_assistant.config import Settings
from course_rag_assistant.web import create_app

app = create_app(settings=Settings(web_data=Path("data/my-course")))
```

`get_settings()` 每次读取新的环境快照，没有全局缓存。Web 应用在创建时固定自己的快照；修改配置后需重启服务，也可以为不同应用实例分别注入配置。显式 `storage` / `--data` 优先于配置中的网页资料路径。

| 环境变量（均以 `COURSE_RAG_` 开头） | 默认值或用途 |
| --- | --- |
| `DATA_DIR` | `data`；统一 CLI 的样例资料默认目录为其下的 `samples` |
| `INDEX_DIR` | `data/index`；CLI 索引和消融评测的默认索引位置 |
| `WEB_DATA` | `data/web`；上传资料与练习记录 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 兼容 API 的聊天服务配置 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` / `OLLAMA_BATCH_SIZE` | 本地嵌入模型；批大小默认 `128` |
| `OLLAMA_CHAT_MODEL` | 本地聊天模型；配置后优先于兼容 API |
| `OLLAMA_CHAT_NUM_PREDICT` | `512` |
| `OLLAMA_CHAT_NUM_CTX` | 默认不传；空字符串也表示不传 |
| `OLLAMA_CHAT_TIMEOUT` | `120` 秒 |
| `OLLAMA_CHAT_THINK` | `false`；接受大小写不敏感的 `true` / `false` |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | 兼容 API 的嵌入服务配置 |
| `RERANK_BASE_URL` / `RERANK_API_KEY` / `RERANK_MODEL` | BGE 重排配置；模型默认 `BAAI/bge-reranker-v2-m3` |

三个目录配置相互独立，修改 `DATA_DIR` 不自动迁移已有索引或网页资料。固定评测脚本仍保留其显式数据路径，以免环境设置改变历史评测输入。

环境中的数值必须为正数，超时必须有限，布尔值必须有效；无效配置会报告变量名，不在错误消息中回显值。密钥字段不会出现在 `repr(Settings)` 中。这里没有自动读取 `.env` 的行为，仍通过环境变量或显式参数配置。

兼容原有服务选择：本地 Ollama 优先；未配置聊天模型时使用离线回答。嵌入服务只填部分必需字段仍报错；聊天和重排服务配置不足时仍返回未配置状态。调用 `Settings.from_env` 会校验整个环境快照，错误的数值配置会提前失败。

## 验证结果

- 全套 233 项测试通过，新增覆盖配置类型转换、密钥表示、无效值、工厂优先级、配置快照和路径覆盖。
- 所有原有网页与 Quiz 测试通过；新旧静态资源 URL 返回同样内容。
- 固定离线基线两次结果一致，12 个证据完整性案例全部通过。覆盖率仍为 67.78%，参考页引用精度仍为 56.82%，正例零拒答、负例全部拒答、重复原句为零。
- 重构后基线见 [baseline-v12-after-structure.json](../eval/reports/baseline-v12-after-structure.json)。评测代码指纹已改为递归记录，包含新的 Web 子模块。
- 构建 wheel 后从解包目录独立导入，确认四类资源齐全，首页、资源、示例问答及三道 Quiz 均可用；未混入旧 `web.py` 或 `web_assets`。
- CLI / Web 模块入口、Ruff 和差异空白检查通过。本轮没有重新运行聊天模型质量评测，历史模型成绩仍参考 v12 报告。

复现基线：

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_baseline --output eval/reports/baseline-v12-after-structure.json
```

暂缓全包 `core/services/utils` 分层，以及 `eval/`、`scripts/` 路径迁移。`evidence` 和 `facts` 承担答案可用性规则，后续分层时应归入业务逻辑，避免仅按文件大小将其归为通用工具。
