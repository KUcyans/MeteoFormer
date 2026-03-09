#!/bin/bash
# ================================================
# start_training_job.sh
# Read timestamp from file and start training
# ================================================

set -e
mkdir -p logs

# --- NEW: activate conda ----
source /Users/yhjo/miniconda3/etc/profile.d/conda.sh
# source /home/cyan_white_tower/miniconda3/etc/profile.d/conda.sh
conda activate tempestransformer
# ----------------------------

# Kill any previous training instances
echo "🧹 Cleaning up old Training processes..."
pkill -f "python3 Train.py" || true
sleep 1

TRACKER="wandb"   # or mlflow

if [ "$TRACKER" = "mlflow" ]; then
    # Read the timestamp from the file
    if [ ! -f logs/.last_stamp ]; then
        echo "❌ No timestamp file found! Run start_mlflow_server.sh first."
        exit 1
    fi
    STAMP=$(cat logs/.last_stamp)
else
    STAMP=$(date +"%Y%m%d_%H%M%S") 
    echo "$STAMP" > logs/.last_stamp # update the last stamp when using wandb

fi

DATE=${STAMP:0:8}
TIME=${STAMP:9}

mkdir -p logs/${DATE}/${TIME}
PYTHON_STDOUT="logs/${DATE}/${TIME}/${DATE}_${TIME}_stdout.log"


nohup python3 Train.py \
  --gpu 0 \
  --date $DATE \
  --time $TIME \
  --tracker $TRACKER \
  > "$PYTHON_STDOUT" 2>&1 &


echo "🚀 Training started (PID: $!)"
