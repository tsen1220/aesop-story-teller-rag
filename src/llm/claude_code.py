"""Claude CLI integration"""
import subprocess
import json
from typing import Optional


class ClaudeCLI:
    """Claude CLI wrapper for generation"""

    def __init__(self):
        """Initialize Claude CLI"""
        self._check_available()

    def _check_available(self) -> bool:
        """Check if Claude CLI is available"""
        try:
            subprocess.run(
                ["which", "claude"],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            raise RuntimeError("Claude CLI not found. Please install it first.")

    def generate(self, prompt: str, timeout: int = 60) -> Optional[str]:
        """
        Generate response using Claude CLI

        Command: claude -p "{prompt}" --output-format json

        Args:
            prompt: Input prompt
            timeout: Timeout in seconds (default: 60)

        Returns:
            Generated text or None if failed
        """
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"✗ Claude CLI error: {result.stderr}")
                return None

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                # Extract text from JSON response
                if isinstance(data, dict):
                    # Claude CLI returns {"result": "...", "type": "result", ...}
                    return data.get("result", data.get("text", data.get("content", str(data))))
                return str(data)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                return result.stdout.strip()

        except subprocess.TimeoutExpired:
            print(f"✗ Claude CLI timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"✗ Claude CLI error: {e}")
            return None


if __name__ == '__main__':
    # Test Claude CLI
    print("Testing Claude CLI...\n")

    try:
        claude = ClaudeCLI()
        response = claude.generate("Tell me a very short story about honesty in one sentence.")

        if response:
            print(f"✓ Claude response:\n{response}\n")
        else:
            print("✗ Claude failed\n")

    except Exception as e:
        print(f"✗ Error: {e}")
