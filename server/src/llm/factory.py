"""LLM Provider Factory with auto-discovery"""
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional
from .base import BaseLLMProvider


class LLMFactory:
    """Factory for creating and managing LLM providers"""
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _auto_discovered = False
    
    @classmethod
    def _auto_discover_providers(cls):
        """Automatically discover all provider classes in the llm directory"""
        if cls._auto_discovered:
            return
        
        llm_dir = Path(__file__).parent
        
        # Scan all Python files in the llm directory
        for file_path in llm_dir.glob("*_provider.py"):
            if file_path.name == "base.py":
                continue
            
            module_name = file_path.stem
            provider_name = module_name.replace("_provider", "")
            
            try:
                # Import the module
                module = importlib.import_module(f".{module_name}", package="src.llm")
                
                # Find all classes that inherit from BaseLLMProvider
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseLLMProvider) and 
                        obj is not BaseLLMProvider and
                        obj.__module__ == module.__name__):
                        cls._providers[provider_name] = obj
                        print(f"✓ Auto-discovered provider: {provider_name}")
                        break
                        
            except Exception as e:
                print(f"✗ Failed to load provider from {file_path.name}: {e}")
        
        cls._auto_discovered = True
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """
        Manually register a custom LLM provider
        
        Args:
            name: Provider identifier (e.g., 'custom_llm')
            provider_class: Class implementing BaseLLMProvider
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
            
        Returns:
            Initialized provider instance
        """
        # Auto-discover providers on first use
        if not cls._auto_discovered:
            cls._auto_discover_providers()
        
        provider_class = cls._providers.get(provider_name.lower())
        
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )
        
        return provider_class(api_key=api_key, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of registered provider names"""
        if not cls._auto_discovered:
            cls._auto_discover_providers()
        return list(cls._providers.keys())
