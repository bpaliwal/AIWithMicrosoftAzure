# import os
# import mlflow
# from azure.monitor.opentelemetry import configure_azure_monitor
# from langchain_community.callbacks.manager import get_openai_callback
# from src.config import Config

# from .logger import Logger
# from .metrics_collector import MetricsCollector
# from .trace_collector import TraceCollector

# # Initialize singleton logger
# logger = Logger()
# metrics = MetricsCollector()
# tracer = TraceCollector()

# def configure_monitoring():
#     """
#     Sets up Azure Monitor and MLflow tracking.
#     """
#     # Configure Azure Monitor if connection string is available
#     if Config.APPLICATIONINSIGHTS_CONNECTION_STRING:
#         configure_azure_monitor(
#             connection_string=Config.APPLICATIONINSIGHTS_CONNECTION_STRING,
#         )
#         print("Azure Monitor configured.")

#     # Configure MLflow
#     if Config.MLFLOW_TRACKING_URI:
#         mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
#         print(f"MLflow tracking URI set: {Config.MLFLOW_TRACKING_URI}")
        
#         # Enable LangChain Autologging
#         try:
#             from mlflow import langchain
#             langchain.autolog(
#                 log_input_examples=True,
#                 log_model_signatures=True,
#                 log_models=True,
#                 registered_model_name=Config.MLFLOW_EXPERIMENT_NAME
#             )
#             print("MLflow LangChain Autologging enabled.")
#         except ImportError:
#             print("Could not import mlflow.langchain. Autologging disabled.")
#         except Exception as e:
#             print(f"Failed to enable LangChain autologging: {e}")

# # Auto-configure on import if desired, or call explicitly in app.py
# configure_monitoring()


"""
Monitoring Module for Travel Chatbot
Handles Azure Monitor and MLflow integration
"""
import os
import mlflow
from opentelemetry import trace
# from azure.monitor.opentelemetry import configure_azure_monitor
from src.config import Config

# # Configure Azure Monitor if available
# if Config.APPLICATIONINSIGHTS_CONNECTION_STRING:
#     try:
#         configure_azure_monitor(
#             connection_string=Config.APPLICATIONINSIGHTS_CONNECTION_STRING,
#         )
#         print("✅ Azure Monitor configured")
#     except Exception as e:
#         print(f"Warning: Azure Monitor configuration failed: {e}")

# Configure MLflow if available
if Config.MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        print(f"✅ MLflow configured: {Config.MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"Warning: MLflow configuration failed: {e}")