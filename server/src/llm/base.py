"""Base LLM Provider interface"""
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider (if needed)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.api_url = api_url
        self.config = kwargs
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str | Iterator[str]:
        """
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            String response or iterator of response chunks if streaming
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass
    
    def is_available(self) -> bool:
        return self.api_key is not None or not self.requires_api_key() or self.api_url is not None
    
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key"""
        return True
