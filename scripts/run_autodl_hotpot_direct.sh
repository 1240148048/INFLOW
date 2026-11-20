#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOTPOT_TRAIN_JSON="${HOTPOT_TRAIN_JSON:-$ROOT/artifacts/datasets/raw_hotpot/hotpot_train_v1.1.json}"
mkdir -p "$(dirname "$HOTPOT_TRAIN_JSON")"
if [[ ! -s "$HOTPOT_TRAIN_JSON" ]]; then
  echo "[Download] Official HotpotQA training JSON; download progress below"
  if ! curl -fL --connect-timeout 20 --retry 3 --retry-delay 3 \
    -o "${HOTPOT_TRAIN_JSON}.part" \
    https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json; then
    curl -fL --connect-timeout 20 --retry 3 --retry-delay 3 \
      -o "${HOTPOT_TRAIN_JSON}.part" \
      http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
  fi
  mv "${HOTPOT_TRAIN_JSON}.part" "$HOTPOT_TRAIN_JSON"
fi
echo "[Data] Using $HOTPOT_TRAIN_JSON; bypassing HF dataset listing for HotpotQA"
exec bash "$ROOT/scripts/run_autodl_hotpot_all.sh"
