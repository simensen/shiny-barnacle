"""Tests for configuration loading from environment variables."""

import os
from unittest.mock import patch


def test_default_config():
    """Test that ProxyConfig loads with default values."""
    # Import inside test to avoid caching issues
    from toolbridge import ProxyConfig

    config = ProxyConfig()
    assert config.backend_url == "http://localhost:8080"
    assert config.port == 4000
    assert config.host == "0.0.0.0"
    assert config.transform_enabled is True
    assert config.debug is False
    assert config.temperature is None


def test_config_from_env_vars():
    """Test that ProxyConfig reads from environment variables."""
    from toolbridge import ProxyConfig

    env_vars = {
        "TOOLBRIDGE_BACKEND_URL": "http://custom:9999",
        "TOOLBRIDGE_PORT": "5000",
        "TOOLBRIDGE_HOST": "127.0.0.1",
        "TOOLBRIDGE_TRANSFORM_ENABLED": "false",
        "TOOLBRIDGE_DEBUG": "true",
        "TOOLBRIDGE_TEMPERATURE": "0.7",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = ProxyConfig()
        assert config.backend_url == "http://custom:9999"
        assert config.port == 5000
        assert config.host == "127.0.0.1"
        assert config.transform_enabled is False
        assert config.debug is True
        assert config.temperature == 0.7


def test_config_partial_env_vars():
    """Test that ProxyConfig uses defaults for unset env vars."""
    from toolbridge import ProxyConfig

    env_vars = {
        "TOOLBRIDGE_BACKEND_URL": "http://partial:8000",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        # Clear any cached env vars that might interfere
        config = ProxyConfig()
        assert config.backend_url == "http://partial:8000"
        assert config.port == 4000  # Default
        assert config.host == "0.0.0.0"  # Default


def test_config_sampling_params():
    """Test that sampling parameters can be set via env vars."""
    from toolbridge import ProxyConfig

    env_vars = {
        "TOOLBRIDGE_TEMPERATURE": "0.5",
        "TOOLBRIDGE_TOP_P": "0.9",
        "TOOLBRIDGE_TOP_K": "40",
        "TOOLBRIDGE_MIN_P": "0.05",
        "TOOLBRIDGE_REPEAT_PENALTY": "1.1",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = ProxyConfig()
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.min_p == 0.05
        assert config.repeat_penalty == 1.1


def test_config_cors_settings():
    """Test CORS configuration via env vars."""
    from toolbridge import ProxyConfig

    env_vars = {
        "TOOLBRIDGE_CORS_ENABLED": "true",
        "TOOLBRIDGE_CORS_ALL_ROUTES": "true",
        "TOOLBRIDGE_CORS_ORIGINS": '["http://localhost:3000", "http://localhost:5173"]',
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = ProxyConfig()
        assert config.cors_enabled is True
        assert config.cors_all_routes is True
        assert config.cors_origins == ["http://localhost:3000", "http://localhost:5173"]
