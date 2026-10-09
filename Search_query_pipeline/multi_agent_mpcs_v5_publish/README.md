# Multi-Agent MPCS v5

用于生成、扩展和验证多跳网页研究问题的多智能体工作流：

`Seed → 事实核验 → Root 约束 → Local 扩展 → Question 结构检查 → Solver / 歧义审查 / Repair`

这是 v5p1 的公开发布副本。Python 包名 `browsecomp_v2` 和部分脚本名沿用历史命名。`hermes/` 内附实际使用的、包含本地修改的 Hermes 运行源码，运行不再依赖作者机器上的外部 Hermes 目录。发布副本已移除历史报告、旧评测、测试套件和开发维护资料；运行需要的技能说明、提示词、插件配置、依赖清单、资源及许可证保留。

**模板不含可用密钥或私有服务地址。真实运行前必须自行配置模型和联网服务。原始运行数据、评测数据、登录凭据及 Python 虚拟环境未包含在仓库中。**

## 目录

| 路径 | 用途 |
| --- | --- |
| `browsecomp_v2/` | 工作流、提示词、验证器、运行器 |
| `hermes/` | Hermes 0.17.0 修改版源码、原始许可证、依赖清单和锁文件 |
| `scripts/` | 完整工作流、分阶段执行及结果导出工具 |
| `start_v5p1.sh` | 使用仓库内 Hermes 的启动器 |
| `.env_v5.example` | v5 工作流配置模板 |
| `config/` | Hermes 配置和服务密钥模板 |
| `examples/sample_seed.json` | 仅供离线 dry-run 使用的虚构 seed |
| `workflow_viewer.html` | 本地查看工作流结果 |
| `data/`、`.hermes/` | 运行时生成，已被 Git 忽略 |

## 1. 安装

建议使用 Linux、Bash 和 Python 3.11。内附 Hermes 声明支持 Python `>=3.11,<3.14`；启动器和进程管理以 Linux 为验证环境。从仓库根目录执行：

```bash
python3.11 -m venv hermes/.venv
hermes/.venv/bin/python -m pip install -e ./hermes
```

依赖定义在 `hermes/pyproject.toml`，同时保留 `hermes/uv.lock`。如已安装 uv，可以用以下命令替代以上安装步骤，按锁文件建立 `hermes/.venv`：

```bash
uv sync --project hermes --locked
```

不要复制其他机器的 `.venv`。pip 方式未锁定所有传递依赖；需要更严格的依赖复现时使用保留的 uv 锁文件。

默认示例使用 Web 服务，不启用浏览器工具。若自行启用浏览器工具，需要 Node.js 20+、`agent-browser` 和相应浏览器依赖，可通过 Hermes 工具配置入口检查；其他可选工具也可能需要额外安装。

## 2. 配置模型和联网服务

```bash
cp .env_v5.example .env_v5
mkdir -p .hermes
cp config/hermes-config.example.yaml .hermes/config.yaml
cp config/.env.hermes.example .hermes/.env
chmod 700 .hermes
chmod 600 .env_v5 .hermes/config.yaml .hermes/.env
```

编辑 `.env_v5`，至少填写：

```dotenv
DEFAULT_API_KEY=填写你的模型服务密钥
DEFAULT_BASE_URL=填写你的模型服务API地址
DEFAULT_MODEL=填写该服务支持的模型名称
DEFAULT_PROVIDER=custom
DEFAULT_BACKEND=hermes
DEFAULT_API_MODE=chat_completions
```

中文文字仅为说明，不是有效配置。使用支持工具调用的模型及与之匹配的 API 模式；默认模板按兼容 Chat Completions 的服务配置。空白的 `SEED_MODEL`、`CONSTRAINT_MODEL`、`SOLVER_MODEL` 等会回退到 `DEFAULT_MODEL`。多模型实验可以按角色设置 `*_MODEL`、`*_API_KEY`、`*_BASE_URL`。

默认 Solver 工具集包含 `vision`。若默认模型不支持图像，需为 `.hermes/config.yaml` 中的 `auxiliary.vision` 配置可用视觉模型，或明确移除不使用的工具。

工作流 `.env_v5` 的值会覆盖同名进程环境变量；其简单解析器**不支持行尾注释、`export KEY=...`、变量插值或通用的 `~` 展开**，注释请单独成行。Hermes 的 YAML 支持 `${VARIABLE}`，示例通过它引用工作流加载的模型配置，无需重复写入密钥。

### 方案 A：公开服务适配器

默认 `.hermes/config.yaml` 选择内附的 Tavily 适配器，承担搜索和网页提取。在 `.hermes/.env` 中填写自己的 `TAVILY_API_KEY`。该适配器使用 Hermes 核心依赖中的 HTTP 客户端。

### 方案 B：自己的搜索和抓取服务

原实验使用独立的 `search_server` 和 `fetch_server`。如有兼容服务，将 `config/hermes-web-services.example.yaml` 的 `web` 段替换到 `.hermes/config.yaml`，填写自己的服务 URL，并在 `.hermes/.env` 中配置相应密钥。服务端实现未附带，客户端协议见：

- `hermes/plugins/web/search_server/provider.py`
- `hermes/plugins/web/fetch_server/provider.py`

更换检索服务会改变搜索结果、网页文本、工具调用数和题目难度；公开模板不保证复现历史实验结果。

## 3. 检查与运行

### 运行时检查

```bash
./start_v5p1.sh --check
```

检查 Python、Hermes 导入和 YAML 结构；输出的 `aiagent_source` 应指向当前仓库的 `hermes/run_agent.py`。不发送模型请求，**不验证密钥有效性、网络连通性或检索服务可用性**。

### 离线 dry-run

```bash
hermes/.venv/bin/python scripts/run_v2_workflow.py \
  --env .env_v5.example \
  --seed examples/sample_seed.json \
  --out-dir data/runs_dry \
  --dry-run
```

该 dry-run 用虚构 seed 检查程序流程，不调用模型，也不验证事实、检索或题目质量。不要将此 seed 用作生产输入。

### 一个 seed 的真实运行

完成服务配置后：

```bash
./start_v5p1.sh 1 v5_demo
```

此命令产生真实服务调用和费用。默认启动器为 1 个 Seed 并发、3 次 Solver rollout、1 个 Solver 并发；模板中的 Local 并发为 2。确认小规模流程成功后再增加负载：

```bash
SEED_CONCURRENCY=2 SOLVER_CONCURRENCY=2 ./start_v5p1.sh 5 v5_batch
```

启动器把 `SEED_CONCURRENCY`、`SEED_PREFETCH`、`SOLVER_ROLLOUTS`、`SOLVER_CONCURRENCY` 写入临时配置，覆盖 `.env_v5` 对应值，原文件不会被改写。其他工作流参数直接编辑 `.env_v5`。

可通过 `ENV_FILE`、`HERMES_PROJECT_ROOT`、`HERMES_VENV`、`HERMES_HOME`、`PYTHON_BIN` 覆盖配置或运行时位置；建议用绝对路径。默认路径均根据启动脚本所在目录计算。

### 已有 seed 和分阶段运行

```bash
hermes/.venv/bin/python scripts/run_v2_workflow.py \
  --env .env_v5 \
  --seed path/to/verified_seed.json \
  --run-id v5_existing \
  --resume
```

真实 seed 的结构与示例一致，但必须有可验证的目标、答案和来源。分阶段入口为 `run_seed_stage.py`、`run_expand_stage.py`、`run_question_stage.py`，参数以各脚本 `--help` 为准。直接运行 Python 脚本也会默认选择仓库内 Hermes，并使用 `.env_v5` 中的并发设置。

## 4. 管线逻辑

以下对应 `.env_v5.example` 的默认配置：

1. **Seed**：生成或读取目标实体与标准答案，去重后进行两次独立事实核验，检查答案是否有来源支持及答案基数；失败时可修复种子再核验。
2. **Root**：构建核心约束和干扰约束。核心约束分别对应候选集合，要求单条不唯一、全部核心集合的交集只留下目标，并检查事实、来源和歧义。
3. **Local**：递归展开约束中的中间实体，形成多跳约束树；默认最大深度为 4，检查事实支撑及父子来源差异。
4. **Question**：选取树路径生成题目，检查结构、路径和深层线索覆盖，默认最多 220 词。当前 `V2_QUESTION_STRUCTURAL_ONLY=1`，因此跳过出题后的独立语义与唯一性检查，将题面歧义判断交给 Solver 后的审查；Seed 和 Root 的检查仍执行。
5. **Solver / Judge**：默认独立解题 3 次，Solver 不接收标准答案或内部约束树；独立 Judge 判分。成功轨迹的 API 与工具调用数分别取中位数，最低可接受门槛为 API ≥10 且工具调用 ≥20。
6. **歧义审查与 Repair**：对错误答案独立核验，只有替代答案有证据满足题面全部条件，才认定歧义。对歧义或过易题目基于已有树修订并重测；默认最多修订 5 次，其中继续剪除捷径最多 3 次，保留较好的已接受版本。

默认 3 次 Solver 的结果含义：

| 结果 | 状态与含义 |
| --- | --- |
| 0 次答对 | `review:all_wrong`：保留供人工复核，不代表已证明题目无歧义 |
| 1 次答对且工作量达标 | `accepted:hard` |
| 2～3 次答对且工作量达标 | `accepted:high_effort` |
| 有正确答案但工作量不足 | 进入修订；预算耗尽后拒绝 |
| 执行或判分失败 | `verification_failed`，不计作答错 |

启动器的种子数量参数表示投入多少个 seed，不保证得到同等数量的有效题。驱动脚本的“有效题”统计只包括 `accepted:hard`、`review:all_wrong` 中答对次数 ≤1 的结果，包含待人工复核的题目。模型、搜索服务和配置变化都会影响结果。

## 5. 输出和恢复

`--run-id v5_demo` 的主要输出为：

- `data/seeds/generated_v5_demo/`：生成的 seed。
- `data/runs_v5_demo/`：artifact、约束树、问题版本、Solver 结果和轨迹。
- `data/tool_workspaces/`、`data/solver_workspaces/`：启动器管理的工具目录。
- `.hermes/`：Hermes 配置、会话、日志和运行状态。

启动器启用 `--resume`；复用 run ID 可能复用已完成结果，独立实验请用新 ID。只删除确认不再需要的运行目录。

## 6. 资源、权限和发布注意事项

- **费用与时长**：低并发不等于低总调用量。验证、重试、Repair 和子代理会增加请求。模板保留 Solver 最多 200 次迭代、单次 7200 秒的预算；服务端费用上限需自行设置。
- **内存**：模板保留至少 6144 MB 可用内存、每调用 1024 MB 的准入预留。小内存容器可能等待；先降低并发，再结合实际资源调整保护参数，不要只为消除等待而关闭保护。
- **执行权限**：`terminal`、`file`、`code_execution` 可执行代码或访问运行账户可见文件，工具工作目录不是操作系统沙箱。建议使用独立容器或专用账户，避免挂载个人凭据和无关数据。
- **日志与数据**：默认保存对话、网页内容和工具输出，可能含敏感信息或第三方材料。日志脱敏不能覆盖所有返回内容，对外分享前需单独审查，不要强制添加被忽略的目录。
- **配置隔离**：默认使用项目自己的 `.hermes/`，不需要复制作者或已有 Hermes 的登录态。API 密钥仅保存在本机私有配置中。
- **结果处理**：`scripts/audit_and_export_non_repeated_v49.py` 需显式提供外部 `--repeat-audit-script`，主流程不依赖它。
- **许可证**：Hermes 的 MIT 许可证及组件署名已保留，见下文。第一方工作流尚未指定许可证，需权利人确认归属和许可后再公开发布。
- **脱敏边界**：已移除原部署地址、个人路径、用户配置和认证状态，具体用户名的示例路径已替换为通用示例。回环地址、容器服务账号、通用系统路径、公开服务域名和必要作者署名保留。重新运行后生成的数据需单独审查。

## 7. 验证范围

发布副本提供 `--check` 与离线 dry-run 作为运行检查入口，不附带历史验证报告或开发测试套件。

真实模型与搜索服务的端到端验证需要填写自己的配置后执行。离线测试通过不代表模型质量或历史实验结果已复现。

## 8. 第三方来源与许可

`hermes/` 基于 [NousResearch/Hermes-Agent](https://github.com/NousResearch/Hermes-Agent) 0.17.0，原源码基线提交为 `1456f09e46bd842e7958e29c315681cddfe276f0`，并包含本地修改，不是未经修改的上游发行版。本地修改涉及辅助模型、响应传输、网页搜索与提取、BrowseComp 约束及请求轨迹捕获。

本发布副本进一步移除了开发文档、历史评测、测试、网站展示内容和发布维护信息，并将个人示例路径通用化。Hermes 原始 MIT 许可证及版权声明见 [hermes/LICENSE](hermes/LICENSE)，其他组件许可证保留在相应目录中。Hermes 的许可证不自动适用于独立的第一方工作流代码。

依赖清单与锁文件、技能及插件运行资源保留。`hermes/website/static/api/model-catalog.json` 是代码读取的模型目录资源，因此保留。独立搜索和抓取服务只附带客户端适配器，不包含其服务端实现或凭据。
