
#!/bin/bash
# Restart MLflow Server Script

echo "Restarting MLflow Server..."

# Step 1: Kill existing MLflow processes
echo "Stopping existing MLflow processes..."
pkill -f "mlflow server" || echo "No existing MLflow process found"
sleep 2

# Step 2: Start fresh MLflow server
echo "Starting MLflow server on http://localhost:5000..."
nohup mlflow server --host 127.0.0.1 --port 5000 > mlflow.log 2>&1 &

# Wait a moment for server to start
sleep 3

# Check if server is running
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow server started successfully!"
    echo "   Dashboard: http://localhost:5000"
    echo "   Logs: mlflow.log"
else
    echo "⚠️ MLflow server may still be starting. Check mlflow.log for details."
fi