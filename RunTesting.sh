#!/bin/bash
# =======================================================
# RunTesting.sh
# Evaluate all checkpoints of a given training run
# =======================================================

set -e
mkdir -p logs

# ---- activate conda ----
# source /Users/yhjo/miniconda3/etc/profile.d/conda.sh
source /home/cyan_white_tower/miniconda3/etc/profile.d/conda.sh
conda activate tempestransformer
# -------------------------

# Kill previous testing processes
echo "🧹 Cleaning up old Test.py processes..."
pkill -f "python3 Test.py" || true
sleep 1

# -------------------------------
# MANUAL SELECTION OF TRAINING RUN
# -------------------------------
DATE=20260529   # training run date: YYYYMMDD
TIME=174259     # training run time: HHMMSS
# -------------------------------

# Timestamp for THIS test run
RUN_DATE=$(date +%Y%m%d)
RUN_TIME=$(date +%H%M%S)

echo "🧪 Testing checkpoints from training run: ${DATE} ${TIME}"
echo "🕒 Test run timestamp: ${RUN_DATE} ${RUN_TIME}"

mkdir -p logs/${DATE}/${TIME}

TEST_STDOUT="logs/${DATE}/${TIME}/${RUN_DATE}_${RUN_TIME}_test.log"


nohup python3 Test.py \
  --date $DATE \
  --time $TIME \
  > "$TEST_STDOUT" 2>&1 &

echo "🚀 Testing started (PID: $!)"
echo "${DATE}"
echo "${TIME}"
