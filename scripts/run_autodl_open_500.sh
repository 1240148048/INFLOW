#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/configs/open_mixed_autodl.json}"
export MODEL="${MODEL:?Set MODEL to the local directory of one of the four backbones}"
MODEL_TAG="$("${PYTHON:-python}" -c 'from pathlib import Path; import re,sys; print(re.sub(r"[^A-Za-z0-9_-]","_",Path(sys.argv[1]).name))' "$MODEL")"
case "${OPEN_MIX_RECIPE:-four_task}" in
  four_task) RECIPE_SUFFIX="" ;;
  mbpp_coqa_squad) RECIPE_SUFFIX="_mbpp200_coqa150_squad150" ;;
  hotpot_coqa_squad) RECIPE_SUFFIX="_hotpot_medium200_coqa150_squad150" ;;
  *) echo "Unknown OPEN_MIX_RECIPE: $OPEN_MIX_RECIPE" >&2; exit 2 ;;
esac
export OUTPUT="${OUTPUT:-$ROOT/artifacts/inflow_${MODEL_TAG}_seed${SEED:-42}_paper_v3_curve${RECIPE_SUFFIX}}"
export UPDATES="${UPDATES:-500}"
export CURVE="${CURVE:-1}"
export DATA_DIR="${DATA_DIR:-$ROOT/artifacts/datasets/inflow_open_mixed_500_seed${SEED:-42}${RECIPE_SUFFIX}}"
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
export TOKENIZERS_PARALLELISM=false
"${PYTHON:-python}" -c 'import torch; from packaging.version import Version; assert Version(torch.__version__.split("+")[0]) >= Version("2.4"), "PyTorch >=2.4 required; keep the 2.5.1 image environment"; assert torch.cuda.is_available(), "CUDA is not available in this Python environment"; print(torch.__version__,torch.cuda.get_device_name(0))'
exec bash "$ROOT/scripts/run_open_tasks.sh" mixed
