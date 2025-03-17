import pytest
import asyncio
from app import verify_model_health, load_tool_configs, get_ollama_models

@pytest.mark.asyncio
async def test_verify_model_health():
    # Test with a known model
    result = await verify_model_health("llama2")
    assert isinstance(result, bool)

def test_load_tool_configs():
    configs = load_tool_configs()
    assert isinstance(configs, list)
    for config in configs:
        assert "type" in config
        assert "function" in config

def test_get_ollama_models():
    models = get_ollama_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert isinstance(models[0], str)
