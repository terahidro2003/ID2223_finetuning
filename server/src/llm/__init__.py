import os
from typing import Dict
from .base import BaseLLMProvider
from .factory import LLMFactory

__all__ = [
    "BaseLLMProvider",
    "LLMFactory",
]


def initialize_providers() -> Dict:
    available_providers = {}
    
    # Auto-discover all providers
    provider_names = LLMFactory.get_available_providers()
    
    for provider_name in provider_names:
        try:
            if provider_name == "mock":
                # Mock provider - always available
                provider = LLMFactory.create_provider(provider_name, delay=0.05)
                if provider.is_available():
                    available_providers[provider.name] = provider

            elif provider_name == "ollama":
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                provider = LLMFactory.create_provider(provider_name, api_url=ollama_url)
                if provider.is_available():
                    available_providers[provider.name] = provider

            else:
                # Generic provider - try to load with API key or URL from env
                env_url = f"{provider_name.upper()}_API_URL"
                env_key = f"{provider_name.upper()}_API_KEY"
                if api_key := os.getenv(env_key):
                    provider = LLMFactory.create_provider(provider_name, api_key=api_key)

                elif api_url := os.getenv(env_url):
                    provider = LLMFactory.create_provider(provider_name, api_url=api_url)

                if provider.is_available():
                    available_providers[provider.name] = provider
                        
        except Exception as e:
            print(f"Failed to initialize {provider_name}: {e}")
    
    return available_providers
