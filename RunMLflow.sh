#!/bin/bash
# ================================================
# start_mlflow_server.sh (safe version)
# ================================================

set -e
mkdir -p logs

# Kill any previous MLflow instances (both UI and server)
echo "🧹 Cleaning up old MLflow processes..."
pkill -f "mlflow ui" || true
pkill -f "mlflow.server.fastapi_app" || true
sleep 1

# Create a new timestamp
STAMP=$(date +"%Y%m%d_%H%M%S")
echo "$STAMP" > logs/.last_stamp

echo "🕒 Timestamp set: $STAMP"
echo "📁 MLflow log: logs/${STAMP}_mlflow.log"

# Start MLflow in background
nohup mlflow ui --port 5000 > logs/${STAMP}_mlflow.log 2>&1 &

echo "🚀 MLflow UI started on port 5000"
echo "   Log file: logs/${STAMP}_mlflow.log"
echo "   Open: http://127.0.0.1:5000"
