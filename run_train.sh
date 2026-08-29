#!/bin/bash
# Train TSMM (requires GPU). Optional: --config path/to/train.yaml
set -e
cd "$(dirname "$0")"
python3 train_best.py "$@"
