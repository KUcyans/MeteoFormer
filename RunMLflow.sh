#!/bin/bash
# ================================================
# start_mlflow_server.sh (safe version)
# ================================================

set -e
mkdir -p logs

# --- NEW: activate conda ----
source /Users/yhjo/miniconda3/etc/profile.d/conda.sh
conda activate tempestransformer
# ----------------------------

# Kill any previous MLflow instances (both UI and server)
echo "🧹 Cleaning up old MLflow processes..."
pkill -f "mlflow ui" || true
pkill -f "mlflow.server.fastapi_app" || true
sleep 1

# Create a new timestamp
STAMP=$(date +"%Y%m%d_%H%M%S")
echo "$STAMP" > logs/.last_stamp

echo "🕒 Timestamp set: $STAMP"
DATE=${STAMP:0:8}
mkdir -p logs/${DATE}

MLFLOW_LOG="logs/${DATE}/${STAMP}_mlflow.log"

echo "📁 MLflow log: $MLFLOW_LOG"

nohup mlflow ui --port 5000 > $MLFLOW_LOG 2>&1 &


echo "🚀 MLflow UI started on port 5000"
echo "   Log file: logs/${STAMP}_mlflow.log"
echo "   Open: http://127.0.0.1:5000"
