
import requests
import json
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class TyphoonTranslateClient:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        """
        Initialize the Typhoon Translate client
        
        Args:
            base_url: vLLM server URL (e.g., "http://your-server-ip:8080")
            api_key: API key if authentication is enabled
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def translate(
        self, 
        text: str, 
        target_language: str = "Thai",
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Translate text using the vLLM server
        
        Args:
            text: Text to translate
            target_language: "Thai" or "English"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Translated text
        """
        system_prompt = f"Translate the following text into {target_language}."
        
        # Format as chat messages (Typhoon Translate expects this format)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        payload = {
            "model": "typhoon-translate",  # Must match --served-model-name
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Translation request failed: {e}")
    
    def batch_translate(
        self,
        texts: List[str],
        target_language: str = "Thai",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_workers: int = 5
    ) -> List[str]:
        """
        Translate multiple texts concurrently
        
        Args:
            texts: List of texts to translate
            target_language: "Thai" or "English"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_workers: Number of concurrent requests
        
        Returns:
            List of translated texts
        """
        def translate_single(text):
            return self.translate(text, target_language, temperature, max_tokens)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(translate_single, texts))
        
        return results