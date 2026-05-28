#!/bin/bash
# ============================================================
# EvoTeam Deploy — Quick Setup for PPU Platform
# Environment: pytorch2.9.0-ubuntu24.04-sdk2.0.0-cuda12.9-vllm0.15.0-sglang0.5.7-py312
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "EvoTeam Deploy — Environment Check"
echo "============================================"

# 1. Python version
echo ""
echo "[1/5] Python version..."
python3 --version

# 2. Dependencies (all pre-installed on platform, just verify)
echo ""
echo "[2/5] Checking dependencies..."
python3 -c "
from evoteam.config import EvoTeamConfig, load_config
from evoteam.models import LocalModel
from evoteam.memory import TaggedCache, VectorCacheStore
from evoteam.evolution import GeneticOptimizer, StrategyChromosome, TieredEvolutionOptimizer
from evoteam.agents import FingerprintAgent, AdaptiveFingerprintAgent
from evoteam.sandbox.sandbox_executor import SandboxExecutor
import openai, yaml, numpy, tiktoken
print('  All imports OK')
"

# 3. Sandbox timeout check
echo ""
echo "[3/5] Checking sandbox timeout..."
python3 -c "
from evoteam.sandbox.sandbox_executor import SandboxExecutor
s = SandboxExecutor(timeout=2)
r = s.execute('while True: pass', 'test', {})
assert not r['success'] and r.get('sandbox_violation') == 'timeout'
print('  Sandbox timeout OK')
"

# 4. Config
echo ""
echo "[4/5] Checking config..."
python3 -c "
from evoteam.config import load_config
c = load_config('./config.yaml')
print(f'  Attack: {c.attack_model}')
print(f'  Target: {c.target_model}')
print(f'  Judge:  {c.judge_model}')
print(f'  Embed:  {c.embedding_model}')
print(f'  API:    {c.api_base}')
"

# 5. vLLM connectivity (optional — models may not be started yet)
echo ""
echo "[5/5] Checking vLLM API..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  vLLM API reachable at http://localhost:8000"
    curl -s http://localhost:8000/v1/models | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m.get('id', '?') for m in data.get('data', [])]
print(f'  Available models: {models}')
"
else
    echo "  vLLM API not reachable yet — start models first"
    echo "  Expected: vllm serve <model> --port 8000"
fi

echo ""
echo "============================================"
echo "Environment check complete."
echo ""
echo "Next steps:"
echo "  1. Start target model:  vllm serve <target-model> --port 8000"
echo "  2. Start attack model:  vllm serve <attack-model> --port 8001"
echo "  3. Start judge model:   vllm serve <judge-model> --port 8002"
echo "  4. Start embed model:   vllm serve <embed-model> --port 8003 --task embedding"
echo "  5. Run: python3 main.py --query 'your query' --verbose"
echo "============================================"
