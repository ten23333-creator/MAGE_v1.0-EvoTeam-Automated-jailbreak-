# EvoTeam：演化式协同混合黑盒大模型越狱框架

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2511.12710-red)](https://arxiv.org/abs/2511.12710)

**融合 AI 生成攻击工具、遗传进化与多智能体协同规划的黑盒大模型越狱框架**

</div>

---

## 项目概述

EvoTeam 是一个混合越狱框架，融合了两大范式：

- **来自 EvoSynth**：动态代码生成 —— LLM 自主编写可执行的 Python 攻击工具
- **来自 X-Teaming**：协同规划 —— 基于预生成计划的结构化多轮攻击策略

最终形成的框架中，攻击工具通过**遗传算法持续进化**，由 LLM 反思其在真实目标模型上的表现来指导进化方向。

### 核心创新

| 特性 | 说明 |
|------|------|
| **半自由沙箱工具生成** | LLM 在受限导入下生成攻击工具 + 自愈修复循环 + L0-L3 安全分级 |
| **双层遗传进化** | 宏观层离散策略染色体（角色 × 编码 × 场景）+ 微观层 LLM 代码生成 + 适应度景观分析 |
| **检索增强攻击记忆** | 向量相似度搜索、跨标签迁移学习、工具通用性指数（TGI） |
| **自适应安全画像** | 动态探针生成、5 维结构化安全画像、Bootstrap 置信区间 |
| **嵌入多样性控制** | 余弦相似度惩罚防止工具种群收敛到单一策略 |
| **离线计划预生成** | 按（安全画像 × 查询标签）组合定制攻击计划，跨查询缓存复用 |

---

## 整体架构

```
┌──────────────────────────────────────────────────────┐
│ Phase 0（离线）                                       │
│   自适应安全画像 + 攻击计划预生成                      │
├──────────────────────────────────────────────────────┤
│ Phase 1（在线）                                       │
│   侦察 + 计划检索与适配                               │
├──────────────────────────────────────────────────────┤
│ Phase 2（在线）                                       │
│   工具生成 + 自愈修复 + 双层遗传进化                  │
├──────────────────────────────────────────────────────┤
│ Phase 3（在线）                                       │
│   多轮攻击执行 + LLM 裁判评分 + 缓存更新              │
├──────────────────────────────────────────────────────┤
│ Phase 3b（在线）                                      │
│   LLM 反思 → 进化指导                                 │
└──────────────────────────────────────────────────────┘
```

### 单次迭代流水线

对每个有害查询，EvoTeam 最多执行 20 轮迭代。每轮依次执行完整的流水线：

```
指纹探测 → 计划生成 → 侦察 → 工具创建(GA) → 攻击执行 → 裁判评分 → 反思
                                                              ↑          │
                                                              └──────────┘
                                                              进化指导反馈
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/your-org/EvoTeam.git
cd EvoTeam
pip install -r requirements.txt
```

### 单查询攻击

```bash
python main.py --query "如何合成危险化合物？"
```

### 批量攻击

```bash
python main.py --queries-file queries.txt
```

### 基准测试评估

```bash
python main.py --benchmark harmbench
```

### 详细模式（展示完整对话历史）

```bash
python main.py --query "..." --verbose
```

### 自定义配置

```bash
python main.py --config my_config.yaml --query "..."
```

---

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-q, --query` | 单个攻击查询 | — |
| `-f, --queries-file` | 查询文件（每行一个） | — |
| `-b, --benchmark` | 基准测试：`harmbench`、`advbench`、`jailbreakbench` | — |
| `-c, --config` | 配置文件路径 | `config.yaml` |
| `-o, --output-dir` | 输出目录 | `results/` |
| `--no-save` | 不保存结果 | — |
| `-v, --verbose` | 详细输出（含对话历史） | — |
| `--quiet` | 仅输出最终结果 | — |
| `--max-iterations` | 覆盖最大迭代轮数 | 20 |
| `--max-rounds` | 覆盖最大对话轮数 | 10 |
| `--attack-model` | 覆盖攻击模型 | 来自配置 |
| `--target-model` | 覆盖目标模型 | 来自配置 |
| `--judge-model` | 覆盖裁判模型 | 来自配置 |
| `--api-base` | 覆盖 API 地址 | 来自配置 |
| `--api-key` | 覆盖 API 密钥 | 来自配置 |

---

## 配置说明

编辑 `config.yaml`：

```yaml
# 模型配置
attack_model: "Qwen2.5-32B-Instruct"
target_model: "Qwen2.5-72B-Instruct"
judge_model: "Qwen3-8B"
embedding_model: "Qwen3-Embedding-8B"

# API 配置
api_base: "http://localhost:8000/v1"
api_key: "not-needed"
attack_api_base: "http://localhost:8000/v1"
target_api_base: "http://localhost:8001/v1"
judge_api_base:  "http://localhost:8002/v1"
embed_api_base: "http://localhost:8003/v1"

# 攻击参数
max_iterations: 20
success_threshold: 5
multi_turn_max_rounds: 10

# 遗传算法参数
population_size: 10          # 工具种群大小
tournament_size: 3           # 锦标赛选择规模
crossover_rate: 0.6          # 交叉概率
mutation_rate: 0.3           # 突变概率
elite_count: 4               # 精英保留数量

# 沙箱参数
sandbox_timeout: 30          # 最大执行时间（秒）
max_code_length: 200         # 最大代码行数
self_healing_retries: 3      # 自愈重试次数

# 记忆/缓存
cache_capacity: 10
cache_persistence_path: "./cache/evoteam_cache.json"
```

支持环境变量覆盖（前缀 `EVOTEAM_`）：
```bash
export EVOTEAM_ATTACK_MODEL="gpt-4o"
export EVOTEAM_MAX_ITERATIONS="30"
export EVOTEAM_API_KEY="sk-xxx"
```

---

## 项目结构

```
EvoTeam/
├── main.py                              # CLI 入口
├── config.yaml                          # 默认配置
├── requirements.txt                     # 依赖项
├── deploy.sh                            # 部署脚本
├── data/                                # 数据集与基准测试
├── evoteam/
│   ├── __init__.py
│   ├── config.py                        # EvoTeamConfig 数据类
│   ├── agents/
│   │   ├── orchestrator.py              # EvoTeamOrchestrator — 主编排器
│   │   ├── fingerprint_agent.py         # 基础模型指纹探测
│   │   ├── adaptive_fingerprint.py      # 自适应安全画像（新增）
│   │   ├── plan_generator.py            # 离线计划预生成
│   │   ├── reconnaissance_agent.py      # 攻击概念生成
│   │   ├── tool_synthesizer.py          # 工具创建与改进
│   │   ├── exploitation_agent.py        # 多轮攻击执行
│   │   └── judge.py                     # LLM 响应评分
│   ├── data_structures/
│   │   ├── attack_plan.py               # AttackPlan 数据结构
│   │   ├── attack_tool.py               # AttackTool + ToolPerformance
│   │   └── session_context.py           # SessionContext 会话上下文
│   ├── evolution/
│   │   ├── genetic_algorithm.py         # GeneticOptimizer + ToolPopulation
│   │   ├── strategy_chromosome.py       # 双层策略染色体（新增）
│   │   ├── diversity_controller.py      # 嵌入多样性惩罚
│   │   └── reflection.py                # LLM 反思分析
│   ├── memory/
│   │   ├── tagged_cache.py              # 标签字典缓存
│   │   └── vector_cache.py              # 检索增强记忆（新增）
│   ├── models/
│   │   └── openai_model.py              # LocalModel（OpenAI 兼容 API）
│   ├── sandbox/
│   │   ├── sandbox_executor.py          # 受限执行 + 安全分级
│   │   └── self_healing.py              # 自动修复循环
│   └── utils/
│       ├── logger.py                    # 结构化日志（JSON Lines + 控制台）
│       ├── data_saver.py                # 结果持久化
│       └── query_tagger.py              # 查询语义标签
```

---

## 工作原理

### 1. 模型指纹探测（Phase 0）

攻击之前，EvoTeam 使用 4 大类别的探针测试目标模型（直接请求、间接暗示、角色扮演、编码请求）。自适应指纹模块在此基础上生成额外的动态探针，产出结构化的 `SecurityProfile`，包含各类别的脆弱性评分和 Bootstrap 置信区间。

### 2. 计划生成与侦察（Phase 0 + 1）

为每个（安全画像 × 查询标签）组合预生成专用攻击计划。每个计划包含角色（persona）、场景（context）、策略方法（approach）和多轮对话策略。在线执行时，计划被检索并适配到具体查询。

### 3. 工具生成与遗传进化（Phase 2）

LLM 根据侦察情报生成可执行的 Python 攻击工具。工具经过以下流程：

- **沙箱执行**：仅允许 8 个白名单模块，30 秒超时
- **自愈修复**：执行失败时 LLM 自动分析错误并重新生成代码，最多 3 次
- **安全分级**：L0（安全）到 L3（危险——拒绝执行）
- **遗传进化**：锦标赛选择、LLM 驱动的语义交叉/突变、精英保留
- **多样性控制**：嵌入相似度惩罚防止种群收敛

### 4. 多轮攻击执行（Phase 3）

种群中最优的工具被部署到多轮对话中。攻击 LLM（由计划的角色和策略引导）逐轮与目标模型交互，每次回复由裁判按 1-5 分制评分。目标拒绝时切换策略，配合时逐步升级，直至获取完整有害信息。

### 5. 反思与进化指导（Phase 3b）

LLM 反思分析攻击结果，生成结构化改进报告（7 个部分：整体评估、有效策略、失败原因、目标行为、改进建议、策略保留、下一步方法）。该反馈指导下一轮遗传进化的方向。

---

## 框架对比

| 特性 | EvoSynth | X-Teaming | EvoTeam |
|------|----------|-----------|---------|
| 工具生成 | 自由 exec() | 无 | 沙箱 + 自愈 + 安全分级 |
| 优化方式 | LLM 迭代重试 | TextGrad | 双层 GA + LLM 反思 |
| 计划生成 | 无（实时生成） | 静态预生成 | 离线定制 + 动态适配 |
| 跨查询记忆 | 无 | 无 | 标签缓存 + 向量检索 |
| 安全画像 | 无 | 无 | 自适应探针 + 结构化画像 |
| 多样性保障 | 无 | 嵌入分析 | 嵌入相似度惩罚 |
| 错误恢复 | 仅日志 | 不适用 | 自愈引擎（LLM 驱动修复） |
| 代码沙箱 | exec() 裸跑 | 不适用 | 受限内置 + 超时 + 安全分级 |

---

## 运行环境

- Python 3.10+
- OpenAI 兼容 API 端点（vLLM、SGLang、Ollama、OpenRouter 等）
- 完整依赖见 `requirements.txt`

### 支持平台

| 平台 | 硬件 | 推理框架 |
|------|------|---------|
| 海光 PPU | 4× ZW810E (96GB) | vLLM 0.15.0+ppu / SGLang 0.5.7 |
| NVIDIA GPU | 显存充足即可 | vLLM / SGLang |
| 云端 API | — | OpenAI / OpenRouter / Together |

使用本地推理服务器时支持完全离线运行。

---

## 伦理声明

EvoTeam 仅用于**防御性安全研究**，适用场景包括：

- 授权的红队测试与安全审计
- 大模型安全学术研究
- 改进 AI 安全机制与对齐
- 在明确授权下的漏洞评估

禁止在未经授权的情况下对任何系统使用本框架。