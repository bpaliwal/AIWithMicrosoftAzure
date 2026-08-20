from opentelemetry import trace
# from azure.monitor.opentelemetry import configure_azure_monitor
from src.config import Config

class TraceCollector:
    """
    Manages distributed tracing setup.
    """
    def __init__(self, service_name="llm_service"):
        self.tracer = trace.get_tracer(service_name)
        
        # # Configure Azure Monitor if available
        # if Config.APPLICATIONINSIGHTS_CONNECTION_STRING:
        #     configure_azure_monitor(
        #         connection_string=Config.APPLICATIONINSIGHTS_CONNECTION_STRING,
        #     )

    def get_tracer(self):
        return self.tracer

    def start_span(self, name: str):
        return self.tracer.start_as_current_span(name)
