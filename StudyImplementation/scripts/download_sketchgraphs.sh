#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

OUTPUT_DIR="$1"
DATA_URL="https://github.com/AutodeskAILab/SketchGraphs/releases/download/v1.0.0/sketches_dataset.jsonl.gz"

mkdir -p "${OUTPUT_DIR}"

FILENAME="$(basename "${DATA_URL}")"
DEST="${OUTPUT_DIR}/${FILENAME}"

if [[ -f "${DEST}" ]]; then
  echo "Dataset already exists at ${DEST}"
  exit 0
fi

echo "Downloading SketchGraphs dataset to ${DEST}"
curl -L "${DATA_URL}" -o "${DEST}"

echo "Download complete."
