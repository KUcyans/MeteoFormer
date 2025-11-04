#!/bin/bash
# ================================================
# start_training_job.sh
# Read timestamp from file and start training
# ================================================

set -e
mkdir -p logs

# --- NEW: activate conda ----
source /Users/yhjo/miniconda3/etc/profile.d/conda.sh
conda activate tempestransformer
# ----------------------------

# Kill any previous training instances
echo "🧹 Cleaning up old Training processes..."
pkill -f "python Train.py" || true
sleep 1

# Read the timestamp
if [ ! -f logs/.last_stamp ]; then
  echo "❌ No timestamp file found! Run start_mlflow_server.sh first."
  exit 1
fi

STAMP=$(cat logs/.last_stamp)
DATE=${STAMP:0:8}
TIME=${STAMP:9}

echo "🕒 Using timestamp: $STAMP"
mkdir -p logs/${DATE}

PYTHON_STDOUT="logs/${DATE}/${DATE}_${TIME}_stdout.log"

nohup python Train.py \
  --gpu 0 \
  --epochs 20 \
  --date $DATE \
  --time $TIME \
  > "$PYTHON_STDOUT" 2>&1 &

echo "🚀 Training started (PID: $!)"
