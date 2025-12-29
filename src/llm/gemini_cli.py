"""Gemini CLI integration"""
import subprocess
import json
from typing import Optional


class GeminiCLI:
    """Gemini CLI wrapper for generation"""

    def __init__(self, model: str = "pro"):
        """
        Initialize Gemini CLI

        Args:
            model: Gemini model to use (default: pro)
        """
        self.model = model
        self._check_available()

    def _check_available(self) -> bool:
        """Check if Gemini CLI is available"""
        try:
            subprocess.run(
                ["which", "gemini"],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            raise RuntimeError("Gemini CLI not found. Please install it first.")

    def generate(self, prompt: str, timeout: int = 60) -> Optional[str]:
        """
        Generate response using Gemini CLI

        Command: gemini -p "{prompt}" -o json --model pro 2>/dev/null

        Args:
            prompt: Input prompt
            timeout: Timeout in seconds (default: 60)

        Returns:
            Generated text or None if failed
        """
        try:
            result = subprocess.run(
                ["gemini", "-p", prompt, "-o", "json", "--model", self.model],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"✗ Gemini CLI error: return code {result.returncode}")
                return None

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                # Extract text from JSON response
                if isinstance(data, dict):
                    # Gemini CLI returns {"response": "...", "stats": {...}}
                    return data.get("response", data.get("text", data.get("content", str(data))))
                return str(data)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                return result.stdout.strip()

        except subprocess.TimeoutExpired:
            print(f"✗ Gemini CLI timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"✗ Gemini CLI error: {e}")
            return None


if __name__ == '__main__':
    # Test Gemini CLI
    print("Testing Gemini CLI...\n")

    try:
        gemini = GeminiCLI()
        response = gemini.generate("Tell me a very short story about honesty in one sentence.")

        if response:
            print(f"✓ Gemini response:\n{response}\n")
        else:
            print("✗ Gemini failed\n")

    except Exception as e:
        print(f"✗ Error: {e}")
