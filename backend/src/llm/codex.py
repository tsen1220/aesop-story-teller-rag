"""Codex CLI integration"""
import subprocess
import json
from typing import Optional


class CodexCLI:
    """Codex CLI wrapper for generation"""

    def __init__(self):
        """Initialize Codex CLI"""
        self._check_available()

    def _check_available(self) -> bool:
        """Check if Codex CLI is available"""
        try:
            subprocess.run(
                ["which", "codex"],
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["which", "jq"],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            raise RuntimeError("Codex CLI or jq not found. Please install them first.")

    def generate(self, prompt: str, timeout: int = 60) -> Optional[str]:
        """
        Generate response using Codex CLI

        Command: codex exec "{prompt}" --json 2>/dev/null | jq 'select(.type=="item.completed" and .item.type=="agent_message")'

        Args:
            prompt: Input prompt
            timeout: Timeout in seconds (default: 60)

        Returns:
            Generated text or None if failed
        """
        try:
            # Use codex with --json flag and pipe to jq
            codex_process = subprocess.Popen(
                ["codex", "exec", prompt, "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )

            jq_process = subprocess.Popen(
                ["jq", 'select(.type=="item.completed" and .item.type=="agent_message")'],
                stdin=codex_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )

            codex_process.stdout.close()
            output, _ = jq_process.communicate(timeout=timeout)

            if jq_process.returncode != 0:
                print("✗ Codex CLI error")
                return None

            # Parse JSON output
            try:
                data = json.loads(output)
                # Extract message from codex response
                if isinstance(data, dict) and "item" in data:
                    item = data["item"]
                    if "content" in item:
                        # Handle content array
                        content = item["content"]
                        if isinstance(content, list) and len(content) > 0:
                            return content[0].get("text", str(content[0]))
                    return item.get("text", str(item))
                return str(data)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                return output.strip()

        except subprocess.TimeoutExpired:
            print(f"✗ Codex CLI timeout after {timeout}s")
            return None
        except Exception as e:
            print(f"✗ Codex CLI error: {e}")
            return None


if __name__ == '__main__':
    # Test Codex CLI
    print("Testing Codex CLI...\n")

    try:
        codex = CodexCLI()
        response = codex.generate("Tell me a very short story about honesty in one sentence.")

        if response:
            print(f"✓ Codex response:\n{response}\n")
        else:
            print("✗ Codex failed\n")

    except Exception as e:
        print(f"✗ Error: {e}")
