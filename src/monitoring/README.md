# Monitoring Module

This module handles Observability and Tracking for the LLM Application using **Azure Monitor** and **MLflow**.

## Features
- **Azure Monitor Integration**: Sends traces and telemetry to Azure Application Insights.
- **MLflow Tracking**: Logs experiments, parameters, and traces to an MLflow server (Local or Azure ML).
- **LangChain Callback**: Provides a callback handler for automatic tracing of LangChain chains.

## Usage
### Automatic Configuration
Simply import this module in your application entry point (`app.py`):
```python
import src.monitoring # Triggers auto-configuration
```

### Manual Tracing
You can use `mlflow` directly for custom logging:
```python
import mlflow

with mlflow.start_run(run_name="my_task"):
    mlflow.log_param("key", "value")
    # ... logic ...
```

## Configuration
Ensure these environment variables are set in `.env`:
- `APPLICATIONINSIGHTS_CONNECTION_STRING`
- `MLFLOW_TRACKING_URI`
