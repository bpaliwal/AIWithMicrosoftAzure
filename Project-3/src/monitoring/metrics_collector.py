import time
from typing import Dict, Any
import mlflow

class MetricsCollector:
    """
    Collects and logs performance and usage metrics.
    """
    def __init__(self):
        self.metrics = {}

    def track_latency(self, func):
        """Decorator to track function latency"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            func_name = func.__name__
            mlflow.log_metric(f"{func_name}_latency", duration)
            return result
        return wrapper

    def log_token_usage(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Logs token usage to MLflow"""
        total_tokens = prompt_tokens + completion_tokens
        mlflow.log_metric("prompt_tokens", prompt_tokens)
        mlflow.log_metric("completion_tokens", completion_tokens)
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_param("model", model)

    def log_custom_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)