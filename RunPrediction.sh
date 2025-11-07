#!/bin/bash
# ================================================
# start_prediction_job.sh
# manual selection of a training run for prediction
# ================================================

set -e
mkdir -p logs

# --- activate conda ----
source /Users/yhjo/miniconda3/etc/profile.d/conda.sh
conda activate tempestransformer
# ------------------------

# kill previous prediction processes
echo "🧹 Cleaning up old Prediction processes..."
pkill -f "python3 Predict.py" || true
sleep 1

# ---- manual selection here ----
DATE=20251107
TIME=131556
# -------------------------------

echo "🕒 Predicting with timestamp: ${DATE} ${TIME}"
mkdir -p logs/${DATE}/${TIME}

# PRED_STDOUT="logs/${DATE}/${DATE}_${TIME}_predict.log"
PRED_STDOUT="logs/${DATE}/${TIME}/${DATE}_${TIME}_predict.log"

nohup python3 Predict.py \
  --date $DATE \
  --time $TIME \
  > "$PRED_STDOUT" 2>&1 &

echo "🔮 Prediction started (PID: $!)"
echo "📄 Output: $PRED_STDOUT"
