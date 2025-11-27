"""Mock LLM Provider for testing"""
import time
from typing import List, Dict, Iterator, Optional
from .base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing without API calls"""
    
    MODELS = [
        "mock-fast",
        "mock-slow",
        "mock-smart",
    ]
    
    # Mock responses for different scenarios
    RESPONSES = {
        "default": "This is a mock response. I'm a fake LLM provider used for testing. "
                   "I can simulate responses without making real API calls. "
                   "Try asking me different questions to see how the interface works!",
        
        "search": "Based on the search results provided, I can see information about {topic}. "
                  "This is a mock response that pretends to use the search context. "
                  "In a real scenario, a real LLM would analyze and synthesize this information.",
        
        "error": "This is a simulated error response for testing error handling.",
    }
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.delay = kwargs.get("delay", 0.05)  # Delay between chunks for streaming
    
    @property
    def name(self) -> str:
        return "Mock (Testing)"
    
    def requires_api_key(self) -> bool:
        return False
    
    def get_available_models(self) -> List[str]:
        return self.MODELS
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str | Iterator[str]:
        
        # Extract user message
        user_message = ""
        has_search_context = False
        
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"].lower()
                if "search" in user_message: has_search_context = True
            elif msg["role"] == "system" and "search results" in msg["content"].lower():
                has_search_context = True
        
        # Choose response based on context
        if has_search_context:
            # Extract a topic from the user message for more realistic mock
            words = user_message.split()
            topic = " ".join(words[:3]) if len(words) >= 3 else "the topic"
            response = self.RESPONSES["search"].format(topic=topic)
        else:
            response = self.RESPONSES["default"]
        
        # Add some variety based on model
        if "smart" in model:
            response += "\n\n(Using 'smart' model - pretending to be more detailed)"
        elif "fast" in model:
            response += "\n\n(Using 'fast' model - quick response)"
        elif "slow" in model:
            response += "\n\n(Using 'slow' model - simulating slower processing)"
            self.delay = 0.1
        
        if stream:
            return self._stream_response(response)
        else:
            return response
    
    def _stream_response(self, response: str) -> Iterator[str]:
        """Stream response word by word"""
        words = response.split()
        for i, word in enumerate(words):
            # Add space before word except for first word
            chunk = word if i == 0 else f" {word}"
            yield chunk
            time.sleep(self.delay)
