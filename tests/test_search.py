from src.config import Config
import pytest

def test_config_loads():
    """Check if config class exists and attributes are reachable"""
    assert hasattr(Config, 'AZURE_OPENAI_API_KEY')

def test_imports():
    """Test that main modules can be imported"""
    from src.search_engine import TravelSearchEngine
    from src.data_loader import TravelDataLoader
    assert TravelSearchEngine is not None
    assert TravelDataLoader is not None