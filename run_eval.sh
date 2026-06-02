#!/bin/bash
# =============================================================================

# This script evaluates a trained VEPO checkpoint on multiple benchmarks:
#   - Geometry3K (Geo3K)
#   - MathVista
#   - MathVerse
#   - MathVision
#   - WeMath
#   - HalluBench
#
# Prerequisites:
#   1. Download evaluation data (see eval/README.md)
#   2. Merge FSDP checkpoint to HuggingFace format using scripts/model_merger.py
#   3. Set up LLM evaluation API (for MathVista/MathVerse/MathVision/WeMath)



set -x

source activate YOUR_ENV_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export LLM_EVAL_BASE_URL="https://api.openai.com/v1"
export LLM_EVAL_API_KEY="YOUR_API_KEY"
export LLM_EVAL_MODEL="gpt-4o-mini"

export OPENROUTER_MAX_CONCURRENT_REQUESTS=2
export OPENROUTER_MIN_INTERVAL=0.8
export OPENROUTER_MAX_RETRIES=8


HF_MODEL_PATH="YOUR_MODEL_PATH" 


RESULTS_DIR="results/vepo_eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/eval/data"
DATASETS="geo3k,hallubench,mathvista,wemath,mathverse,mathvision"


SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""


cd "${SCRIPT_DIR}/eval"

python main.py \
  --model ${HF_MODEL_PATH} \
  --output-dir ${RESULTS_DIR} \
  --data-path ${DATA_DIR} \
  --datasets ${DATASETS} \
  --tensor-parallel-size 2 \
  --system-prompt="${SYSTEM_PROMPT}" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --eval-threads 24

echo "Evaluation complete. Results saved to: ${RESULTS_DIR}"
echo ""
echo "Results per dataset:"
for dataset in $(echo ${DATASETS} | tr ',' ' '); do
    if [ -f "${RESULTS_DIR}/${dataset}.json" ]; then
        echo "  ${dataset}: $(python -c "import json; d=json.load(open('${RESULTS_DIR}/${dataset}.json')); print(f\"Accuracy={d['metrics']['accuracy']:.4f}\")")"
    fi
done
