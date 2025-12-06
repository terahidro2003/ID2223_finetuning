"""Finetuned LLM Provider"""
from typing import List, Dict, Iterator, Optional
from .base import BaseLLMProvider
from openai import OpenAI
import httpx, json

class FineTomeProvider(BaseLLMProvider):
    """Finetuned on FineTome LLM provider"""
    
    MODELS = {
        "Llama 3.2 1B base": {
            "path": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "client": None
        },
        "Llama 3.2 3B base": {
            "path": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "client": None
        },
        "Llama 3.2 3B lora": {
            "path": "hellstone1918/Llama-3.2-3B-basic-lora-model",
            "client": None
        },
        "Llama 3.2 3B lora finance": {
            "path": "hellstone1918/Llama-3.2-3B-finance-lora-model-v6",
            "client": None
        },
        "Llama 3.2 3B rs lora": {
            "path": "hellstone1918/Llama-3.2-3B-rslora-model",
            "client": None
        },
    }
    
    def __init__(self, api_url: Optional[str] = None, **kwargs):
        super().__init__(api_url=api_url, **kwargs)
        if api_url:

            for url in api_url.split(','):
                for model in self.MODELS:
                    if model.replace(' ', '-').replace('.', '-').lower() in url:
                        base_url=url.strip('/ ') + '/v1/'
                        self.MODELS[model]["client"] = OpenAI(base_url=base_url, api_key="EMPTY TO INITIALIZE WITHOUT FAILURE")
                    
        else:
            self.client = None
    
    @property
    def name(self) -> str:
        return "Finetome LLM"
    
    def requires_api_key(self) -> bool:
        return False
    
    def get_available_models(self) -> List[str]:
        return [model for model in self.MODELS if self.MODELS[model]["client"]]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str | Iterator[str]:
        try:
            response = self.MODELS[model]["client"].chat.completions.create(
                model=self.MODELS[model]["path"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")
    
    def _stream_response(self, response) -> Iterator[str]:
        """Stream response chunks"""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
