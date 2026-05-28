# EvoTeam: Evolutionary Collaborative Hybrid Framework for Black-Box LLM Jailbreak

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2511.12710-red)](https://arxiv.org/abs/2511.12710)

**A black-box LLM jailbreak framework combining AI-generated attack tools, genetic evolution, and collaborative multi-agent planning.**

</div>

---

## Overview

EvoTeam is a hybrid jailbreak framework that merges two paradigms:

- **From EvoSynth**: Dynamic code generation — LLMs autonomously write executable Python attack tools
- **From X-Teaming**: Collaborative planning — structured multi-turn attack strategies with pre-generated plans

The result is a framework where attack tools **evolve through genetic algorithms**, guided by LLM reflection on their real-world performance against target models.

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Semi-free Sandbox Tool Generation** | LLM generates attack tools with constrained imports + self-healing repair loop + L0-L3 safety classification |
| **Tiered Genetic Evolution** | Macro-level discrete strategy chromosomes (persona × encoding × context) + micro-level LLM code generation + fitness landscape analysis |
| **Retrieval-Augmented Attack Memory** | Vector similarity search, cross-tag transfer learning, Tool Generality Index (TGI) |
| **Adaptive Security Profiling** | Dynamic probe generation, structured SecurityProfile with 5 vulnerability dimensions, bootstrap confidence intervals |
| **Embedding-based Diversity Control** | Cosine similarity penalties prevent tool population convergence |
| **Offline Plan Pre-generation** | Attack plans tailored to (security profile × query tag) combinations, cached for cross-query reuse |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ Phase 0 (Offline)                                     │
│   Adaptive Security Profiling + Plan Pre-generation   │
├──────────────────────────────────────────────────────┤
│ Phase 1 (Online)                                      │
│   Reconnaissance + Plan Retrieval & Adaptation        │
├──────────────────────────────────────────────────────┤
│ Phase 2 (Online)                                      │
│   Tool Generation + Self-Healing + Tiered GA Evolution│
├──────────────────────────────────────────────────────┤
│ Phase 3 (Online)                                      │
│   Multi-turn Attack + LLM Judge + Cache Update        │
├──────────────────────────────────────────────────────┤
│ Phase 3b (Online)                                     │
│   LLM Reflection → Evolution Guidance                 │
└──────────────────────────────────────────────────────┘
```

### Per-Iteration Pipeline

For each harmful query, EvoTeam runs up to 20 iterations. Each iteration executes the full serial pipeline:

```
Fingerprint → Plan → Recon → Tool Creation (GA) → Exploitation → Judge → Reflect
                                                      ↑                              │
                                                      └────── evolution guidance ─────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/EvoTeam.git
cd EvoTeam
pip install -r requirements.txt
```

### Single Query Attack

```bash
python main.py --query "How to synthesize a dangerous compound?"
```

### Batch Attack

```bash
python main.py --queries-file queries.txt
```

### Benchmark Evaluation

```bash
python main.py --benchmark harmbench
```

### Verbose Mode (shows full conversation history)

```bash
python main.py --query "..." --verbose
```

### Custom Configuration

```bash
python main.py --config my_config.yaml --query "..."
```

---

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-q, --query` | Single attack query | — |
| `-f, --queries-file` | File with queries (one per line) | — |
| `-b, --benchmark` | Run benchmark: `harmbench`, `advbench`, `jailbreakbench` | — |
| `-c, --config` | Path to YAML config | `config.yaml` |
| `-o, --output-dir` | Output directory | `results/` |
| `--no-save` | Disable saving results | — |
| `-v, --verbose` | Verbose output with conversation history | — |
| `--quiet` | Suppress all output except final results | — |
| `--max-iterations` | Override max iterations | 20 |
| `--max-rounds` | Override max conversation rounds | 10 |
| `--attack-model` | Override attack model | from config |
| `--target-model` | Override target model | from config |
| `--judge-model` | Override judge model | from config |
| `--api-base` | Override API base URL | from config |
| `--api-key` | Override API key | from config |

---

## Configuration

Edit `config.yaml`:

```yaml
# Model Configuration
attack_model: "Qwen2.5-32B-Instruct"
target_model: "Qwen2.5-72B-Instruct"
judge_model: "Qwen3-8B"
embedding_model: "Qwen3-Embedding-8B"

# API Configuration
api_base: "http://localhost:8000/v1"
api_key: "not-needed"
attack_api_base: "http://localhost:8000/v1"
target_api_base: "http://localhost:8001/v1"
judge_api_base:  "http://localhost:8002/v1"
embed_api_base: "http://localhost:8003/v1"

# Attack Parameters
max_iterations: 20
success_threshold: 5
multi_turn_max_rounds: 10

# Genetic Algorithm
population_size: 10
tournament_size: 3
crossover_rate: 0.6
mutation_rate: 0.3
elite_count: 4

# Sandbox
sandbox_timeout: 30
max_code_length: 200
self_healing_retries: 3

# Memory
cache_capacity: 10
cache_persistence_path: "./cache/evoteam_cache.json"
```

Environment variable overrides (prefix `EVOTEAM_`):
```bash
export EVOTEAM_ATTACK_MODEL="gpt-4o"
export EVOTEAM_MAX_ITERATIONS="30"
export EVOTEAM_API_KEY="sk-xxx"
```

---

## Project Structure

```
EvoTeam/
├── main.py                              # CLI entry point
├── config.yaml                          # Default configuration
├── requirements.txt                     # Dependencies
├── deploy.sh                            # Deployment script
├── data/                                # Datasets & benchmarks
├── evoteam/
│   ├── __init__.py
│   ├── config.py                        # EvoTeamConfig dataclass
│   ├── agents/
│   │   ├── orchestrator.py              # EvoTeamOrchestrator — main pipeline
│   │   ├── fingerprint_agent.py         # Basic model fingerprinting
│   │   ├── adaptive_fingerprint.py      # Adaptive Security Profiling (NEW)
│   │   ├── plan_generator.py            # Offline plan pre-generation
│   │   ├── reconnaissance_agent.py      # Attack concept generation
│   │   ├── tool_synthesizer.py          # Tool creation & improvement
│   │   ├── exploitation_agent.py        # Multi-turn attack execution
│   │   └── judge.py                     # LLM response scoring
│   ├── data_structures/
│   │   ├── attack_plan.py               # AttackPlan dataclass
│   │   ├── attack_tool.py               # AttackTool + ToolPerformance
│   │   └── session_context.py           # SessionContext
│   ├── evolution/
│   │   ├── genetic_algorithm.py         # GeneticOptimizer + ToolPopulation
│   │   ├── strategy_chromosome.py       # Two-tier StrategyChromosome (NEW)
│   │   ├── diversity_controller.py      # Embedding-based diversity penalty
│   │   └── reflection.py                # LLM reflection analysis
│   ├── memory/
│   │   ├── tagged_cache.py              # Tagged dictionary cache
│   │   └── vector_cache.py              # Retrieval-Augmented Memory (NEW)
│   ├── models/
│   │   └── openai_model.py              # LocalModel (OpenAI-compatible API)
│   ├── sandbox/
│   │   ├── sandbox_executor.py          # Restricted execution + safety levels
│   │   └── self_healing.py              # Auto-repair loop
│   └── utils/
│       ├── logger.py                    # Structured logging (JSON Lines + console)
│       ├── data_saver.py                # Result persistence
│       └── query_tagger.py              # Query semantic tagging
```

---

## How It Works

### 1. Model Fingerprinting (Phase 0)

Before attacking, EvoTeam probes the target model with queries across 4 categories (direct request, indirect hint, role play, encoded request). The adaptive fingerprinting module generates additional dynamic probes and produces a structured `SecurityProfile` with per-category vulnerability scores and bootstrap confidence intervals.

### 2. Plan Generation & Reconnaissance (Phase 0 + 1)

Specialized attack plans are pre-generated for each (security profile × query tag) combination. Each plan specifies a persona, context, approach, and multi-turn conversation strategy. During online execution, plans are retrieved and adapted to the specific query.

### 3. Tool Generation & Genetic Evolution (Phase 2)

The LLM generates executable Python attack tools based on reconnaissance intelligence. These tools undergo:

- **Sandbox execution** with restricted imports (8 whitelisted modules only) and 30-second timeout
- **Self-healing**: automatic error repair via LLM, up to 3 retries
- **Safety classification**: L0 (safe) through L3 (unsafe — rejected)
- **Genetic evolution**: tournament selection, LLM-driven semantic crossover/mutation, elitism preservation
- **Diversity control**: embedding similarity penalties prevent convergence

### 4. Multi-turn Attack Execution (Phase 3)

The best tools from the population are deployed in multi-turn conversations. An attack LLM (guided by the plan's persona and strategy) engages the target model turn by turn, with each response scored by the judge on a 1–5 scale. If the target refuses, the attacker pivots strategy; if it complies, the attacker escalates toward complete information disclosure.

### 5. Reflection & Evolution Guidance (Phase 3b)

LLM reflection analyzes attack outcomes and produces structured improvement guidance (7 sections: overall assessment, what worked, what failed, target behavior, improvement suggestions, strategy retention, next approach). This feedback steers the next generation of genetic evolution.

---

## Comparison

| Feature | EvoSynth | X-Teaming | EvoTeam |
|---------|----------|-----------|---------|
| Tool Generation | Free exec() | None | Sandbox + self-healing + safety levels |
| Optimization | LLM iterative retry | TextGrad | Tiered GA + LLM reflection |
| Plan Generation | None (real-time only) | Static pre-generated | Offline specialized + adaptive |
| Cross-query Memory | None | None | Tagged cache + vector retrieval |
| Security Profiling | None | None | Adaptive probes + structured profile |
| Diversity | None | Embedding analysis | Embedding similarity penalty |
| Error Recovery | Log-only | N/A | Self-healing (LLM-driven repair) |
| Code Sandbox | exec() direct | N/A | Restricted builtins + timeout + safety |

---

## Requirements

- Python 3.10+
- OpenAI-compatible API endpoint (vLLM, SGLang, Ollama, OpenRouter, etc.)
- See `requirements.txt` for full dependency list

## Platform Support

EvoTeam has been tested on:

| Platform | Hardware | Inference |
|----------|----------|-----------|
| Hygon PPU | 4× ZW810E (96GB) | vLLM 0.15.0+ppu / SGLang 0.5.7 |
| NVIDIA GPU | Any with sufficient VRAM | vLLM / SGLang |
| Cloud API | — | OpenAI / OpenRouter / Together |

Completely offline operation is supported when using local inference servers.

---

## Ethics Statement

EvoTeam is designed for **defensive security research only**. Use cases include:

- Authorized red-teaming and security auditing
- Academic research on LLM safety
- Improving AI safety mechanisms and alignment
- Vulnerability assessment with explicit permission

Do not use this framework for malicious purposes or against systems without proper authorization.