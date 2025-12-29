"""Ollama Python SDK integration for local LLM"""
import ollama
from typing import Optional, List, Dict


class Ollama:
    """Ollama wrapper for local LLM generation"""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize Ollama

        Args:
            model: Model to use (if None, will use first available model)
        """
        self.available_models = self.list_models()

        if model:
            if model not in [m["name"] for m in self.available_models]:
                raise ValueError(f"Model '{model}' not found. Available: {[m['name'] for m in self.available_models]}")
            self.model = model
        elif self.available_models:
            self.model = self.available_models[0]["name"]
        else:
            raise RuntimeError("No models available. Please pull a model first: ollama pull <model>")

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available models from Ollama

        Returns:
            List of available models with name and size info
        """
        try:
            response = ollama.list()
            models = []

            for model in response.models:
                models.append({
                    "name": model.model,
                    "size": self._format_size(model.size),
                    "modified_at": str(model.modified_at),
                    "family": model.details.family if model.details else ""
                })

            return models

        except Exception as e:
            print(f"✗ Ollama list error: {e}")
            return []

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable size"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        for model in self.available_models:
            if model["name"] == self.model:
                return model
        return {"name": self.model, "size": "unknown"}

    def set_model(self, model: str) -> None:
        """
        Set the model to use

        Args:
            model: Model name to use
        """
        if model not in [m["name"] for m in self.available_models]:
            raise ValueError(f"Model '{model}' not found. Available: {[m['name'] for m in self.available_models]}")
        self.model = model

    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate response using Ollama

        Args:
            prompt: Input prompt

        Returns:
            Generated text or None if failed
        """
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )
            return response.get("response", None)

        except Exception as e:
            print(f"✗ Ollama error: {e}")
            return None

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Chat with the model (multi-turn conversation)

        Args:
            messages: List of messages [{"role": "user", "content": "..."}]

        Returns:
            Assistant response or None if failed
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages
            )
            return response.get("message", {}).get("content", None)

        except Exception as e:
            print(f"✗ Ollama chat error: {e}")
            return None


if __name__ == '__main__':
    # Test Ollama
    print("Testing Ollama...\n")

    try:
        oll = Ollama()

        # Show available models
        print("Available models:")
        for model in oll.available_models:
            marker = "→" if model["name"] == oll.model else " "
            print(f"  {marker} {model['name']} ({model['size']})")
        print()

        # Test generation
        response = oll.generate("Tell me a very short story about honesty in one sentence.")

        if response:
            print(f"✓ Ollama ({oll.model}) response:\n{response}\n")
        else:
            print("✗ Ollama failed\n")

    except Exception as e:
        print(f"✗ Error: {e}")
