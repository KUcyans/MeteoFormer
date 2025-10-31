#!/bin/bash
# ================================================
# start_training_job.sh
# Read timestamp from file and start training
# ================================================

set -e
mkdir -p logs

# Read the timestamp
if [ ! -f logs/.last_stamp ]; then
  echo "❌ No timestamp file found! Run start_mlflow_server.sh first."
  exit 1
fi

STAMP=$(cat logs/.last_stamp)
DATE=${STAMP:0:8}
TIME=${STAMP:9}

echo "🕒 Using timestamp: $STAMP"
echo "📁 Training log: logs/${STAMP}_train.log"

nohup python Train.py \
  --gpu 0 \
  --epochs 10 \
  --date $DATE \
  --time $TIME \
  > logs/${STAMP}_train.log 2>&1 &

echo "🚀 Training started (PID: $!)"
echo "   Log file: logs/${STAMP}_train.log"
