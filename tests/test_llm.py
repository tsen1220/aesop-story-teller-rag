"""Unit tests for LLM modules"""

import pytest
from unittest.mock import patch, MagicMock


class TestOllama:
    """Test Ollama class"""

    @patch('src.llm.ollama.ollama')
    def test_init_with_model(self, mock_ollama_sdk):
        """Test Ollama initialization with specific model"""
        from src.llm.ollama import Ollama

        # Mock list response
        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_model.size = 5000000000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = MagicMock()
        mock_model.details.family = "llama"

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama(model="llama3.1:8b")

        assert oll.model == "llama3.1:8b"
        assert len(oll.available_models) == 1

    @patch('src.llm.ollama.ollama')
    def test_init_without_model_uses_first_available(self, mock_ollama_sdk):
        """Test Ollama initialization uses first available model"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "llama3.2:latest"
        mock_model.size = 5000000000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = MagicMock()
        mock_model.details.family = "llama"

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama()

        assert oll.model == "llama3.2:latest"

    @patch('src.llm.ollama.ollama')
    def test_init_model_not_found(self, mock_ollama_sdk):
        """Test Ollama raises error when model not found"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_model.size = 5000000000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        with pytest.raises(ValueError, match="not found"):
            Ollama(model="nonexistent:model")

    @patch('src.llm.ollama.ollama')
    def test_init_no_models_available(self, mock_ollama_sdk):
        """Test Ollama raises error when no models available"""
        from src.llm.ollama import Ollama

        mock_response = MagicMock()
        mock_response.models = []
        mock_ollama_sdk.list.return_value = mock_response

        with pytest.raises(RuntimeError, match="No models available"):
            Ollama()

    @patch('src.llm.ollama.ollama')
    def test_list_models_error(self, mock_ollama_sdk):
        """Test list_models handles exceptions"""
        from src.llm.ollama import Ollama

        mock_ollama_sdk.list.side_effect = Exception("Connection error")

        # This will raise RuntimeError because no models are available
        with pytest.raises(RuntimeError, match="No models available"):
            Ollama()

    @patch('src.llm.ollama.ollama')
    def test_format_size(self, mock_ollama_sdk):
        """Test _format_size method"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "test"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama()

        # Test different sizes
        assert "B" in oll._format_size(100)
        assert "KB" in oll._format_size(2048)
        assert "MB" in oll._format_size(2 * 1024 * 1024)
        assert "GB" in oll._format_size(2 * 1024 * 1024 * 1024)
        assert "TB" in oll._format_size(2 * 1024 * 1024 * 1024 * 1024)

    @patch('src.llm.ollama.ollama')
    def test_get_model_info(self, mock_ollama_sdk):
        """Test get_model_info method"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_model.size = 5000000000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = MagicMock()
        mock_model.details.family = "llama"

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama(model="llama3.1:8b")
        info = oll.get_model_info()

        assert info["name"] == "llama3.1:8b"

    @patch('src.llm.ollama.ollama')
    def test_get_model_info_not_found(self, mock_ollama_sdk):
        """Test get_model_info returns unknown when model not in list"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_model.size = 5000000000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama()
        oll.model = "different:model"  # Force a different model
        info = oll.get_model_info()

        assert info["size"] == "unknown"

    @patch('src.llm.ollama.ollama')
    def test_set_model_success(self, mock_ollama_sdk):
        """Test set_model method"""
        from src.llm.ollama import Ollama

        mock_model1 = MagicMock()
        mock_model1.model = "model1"
        mock_model1.size = 1000
        mock_model1.modified_at = "2024-01-01"
        mock_model1.details = None

        mock_model2 = MagicMock()
        mock_model2.model = "model2"
        mock_model2.size = 2000
        mock_model2.modified_at = "2024-01-01"
        mock_model2.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model1, mock_model2]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama()
        oll.set_model("model2")

        assert oll.model == "model2"

    @patch('src.llm.ollama.ollama')
    def test_set_model_not_found(self, mock_ollama_sdk):
        """Test set_model raises error for unknown model"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "model1"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response

        oll = Ollama()

        with pytest.raises(ValueError, match="not found"):
            oll.set_model("nonexistent")

    @patch('src.llm.ollama.ollama')
    def test_generate_success(self, mock_ollama_sdk):
        """Test generate method"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "test"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response
        mock_ollama_sdk.generate.return_value = {"response": "Test response"}

        oll = Ollama()
        result = oll.generate("Hello")

        assert result == "Test response"
        mock_ollama_sdk.generate.assert_called_once()

    @patch('src.llm.ollama.ollama')
    def test_generate_error(self, mock_ollama_sdk):
        """Test generate handles errors"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "test"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response
        mock_ollama_sdk.generate.side_effect = Exception("Generate error")

        oll = Ollama()
        result = oll.generate("Hello")

        assert result is None

    @patch('src.llm.ollama.ollama')
    def test_chat_success(self, mock_ollama_sdk):
        """Test chat method"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "test"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response
        mock_ollama_sdk.chat.return_value = {"message": {"content": "Chat response"}}

        oll = Ollama()
        result = oll.chat([{"role": "user", "content": "Hello"}])

        assert result == "Chat response"

    @patch('src.llm.ollama.ollama')
    def test_chat_error(self, mock_ollama_sdk):
        """Test chat handles errors"""
        from src.llm.ollama import Ollama

        mock_model = MagicMock()
        mock_model.model = "test"
        mock_model.size = 1000
        mock_model.modified_at = "2024-01-01"
        mock_model.details = None

        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama_sdk.list.return_value = mock_response
        mock_ollama_sdk.chat.side_effect = Exception("Chat error")

        oll = Ollama()
        result = oll.chat([{"role": "user", "content": "Hello"}])

        assert result is None


class TestClaudeCLI:
    """Test ClaudeCLI class"""

    @patch('src.llm.claude_code.subprocess.run')
    def test_init_available(self, mock_run):
        """Test ClaudeCLI initialization when CLI is available"""
        from src.llm.claude_code import ClaudeCLI

        mock_run.return_value = MagicMock(returncode=0)

        cli = ClaudeCLI()
        assert cli is not None

    @patch('src.llm.claude_code.subprocess.run')
    def test_init_not_available(self, mock_run):
        """Test ClaudeCLI raises error when CLI not available"""
        from src.llm.claude_code import ClaudeCLI
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "which")

        with pytest.raises(RuntimeError, match="not found"):
            ClaudeCLI()

    @patch('src.llm.claude_code.subprocess.run')
    def test_generate_success_json(self, mock_run):
        """Test generate with JSON response"""
        from src.llm.claude_code import ClaudeCLI

        # First call for _check_available, second for generate
        mock_run.side_effect = [
            MagicMock(returncode=0),  # which claude
            MagicMock(returncode=0, stdout='{"result": "Test response"}')
        ]

        cli = ClaudeCLI()
        result = cli.generate("Hello")

        assert result == "Test response"

    @patch('src.llm.claude_code.subprocess.run')
    def test_generate_success_raw(self, mock_run):
        """Test generate with raw text response"""
        from src.llm.claude_code import ClaudeCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout='Raw text response')
        ]

        cli = ClaudeCLI()
        result = cli.generate("Hello")

        assert result == "Raw text response"

    @patch('src.llm.claude_code.subprocess.run')
    def test_generate_error(self, mock_run):
        """Test generate handles CLI errors"""
        from src.llm.claude_code import ClaudeCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1, stderr="Error")
        ]

        cli = ClaudeCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.claude_code.subprocess.run')
    def test_generate_timeout(self, mock_run):
        """Test generate handles timeout"""
        from src.llm.claude_code import ClaudeCLI
        import subprocess

        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.TimeoutExpired("claude", 60)
        ]

        cli = ClaudeCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.claude_code.subprocess.run')
    def test_generate_exception(self, mock_run):
        """Test generate handles general exceptions"""
        from src.llm.claude_code import ClaudeCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            Exception("Unknown error")
        ]

        cli = ClaudeCLI()
        result = cli.generate("Hello")

        assert result is None


class TestGeminiCLI:
    """Test GeminiCLI class"""

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_init_available(self, mock_run):
        """Test GeminiCLI initialization when CLI is available"""
        from src.llm.gemini_cli import GeminiCLI

        mock_run.return_value = MagicMock(returncode=0)

        cli = GeminiCLI()
        assert cli is not None
        assert cli.model == "pro"

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_init_not_available(self, mock_run):
        """Test GeminiCLI raises error when CLI not available"""
        from src.llm.gemini_cli import GeminiCLI
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "which")

        with pytest.raises(RuntimeError, match="not found"):
            GeminiCLI()

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_generate_success_json(self, mock_run):
        """Test generate with JSON response"""
        from src.llm.gemini_cli import GeminiCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout='{"response": "Test response"}')
        ]

        cli = GeminiCLI()
        result = cli.generate("Hello")

        assert result == "Test response"

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_generate_success_raw(self, mock_run):
        """Test generate with raw text response"""
        from src.llm.gemini_cli import GeminiCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout='Raw text response')
        ]

        cli = GeminiCLI()
        result = cli.generate("Hello")

        assert result == "Raw text response"

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_generate_error(self, mock_run):
        """Test generate handles CLI errors"""
        from src.llm.gemini_cli import GeminiCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1)
        ]

        cli = GeminiCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_generate_timeout(self, mock_run):
        """Test generate handles timeout"""
        from src.llm.gemini_cli import GeminiCLI
        import subprocess

        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.TimeoutExpired("gemini", 60)
        ]

        cli = GeminiCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.gemini_cli.subprocess.run')
    def test_generate_exception(self, mock_run):
        """Test generate handles general exceptions"""
        from src.llm.gemini_cli import GeminiCLI

        mock_run.side_effect = [
            MagicMock(returncode=0),
            Exception("Unknown error")
        ]

        cli = GeminiCLI()
        result = cli.generate("Hello")

        assert result is None


class TestCodexCLI:
    """Test CodexCLI class"""

    @patch('src.llm.codex.subprocess.run')
    def test_init_available(self, mock_run):
        """Test CodexCLI initialization when CLI is available"""
        from src.llm.codex import CodexCLI

        mock_run.return_value = MagicMock(returncode=0)

        cli = CodexCLI()
        assert cli is not None

    @patch('src.llm.codex.subprocess.run')
    def test_init_not_available(self, mock_run):
        """Test CodexCLI raises error when CLI not available"""
        from src.llm.codex import CodexCLI
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "which")

        with pytest.raises(RuntimeError, match="not found"):
            CodexCLI()

    @patch('src.llm.codex.subprocess.Popen')
    @patch('src.llm.codex.subprocess.run')
    def test_generate_success(self, mock_run, mock_popen):
        """Test generate with successful response"""
        from src.llm.codex import CodexCLI

        mock_run.return_value = MagicMock(returncode=0)

        # Mock codex process
        mock_codex = MagicMock()
        mock_codex.stdout = MagicMock()

        # Mock jq process
        mock_jq = MagicMock()
        mock_jq.communicate.return_value = ('{"item": {"content": [{"text": "Test response"}]}}', "")
        mock_jq.returncode = 0

        mock_popen.side_effect = [mock_codex, mock_jq]

        cli = CodexCLI()
        result = cli.generate("Hello")

        assert result == "Test response"

    @patch('src.llm.codex.subprocess.Popen')
    @patch('src.llm.codex.subprocess.run')
    def test_generate_jq_error(self, mock_run, mock_popen):
        """Test generate handles jq errors"""
        from src.llm.codex import CodexCLI

        mock_run.return_value = MagicMock(returncode=0)

        mock_codex = MagicMock()
        mock_codex.stdout = MagicMock()

        mock_jq = MagicMock()
        mock_jq.communicate.return_value = ("", "Error")
        mock_jq.returncode = 1

        mock_popen.side_effect = [mock_codex, mock_jq]

        cli = CodexCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.codex.subprocess.Popen')
    @patch('src.llm.codex.subprocess.run')
    def test_generate_timeout(self, mock_run, mock_popen):
        """Test generate handles timeout"""
        from src.llm.codex import CodexCLI
        import subprocess

        mock_run.return_value = MagicMock(returncode=0)

        mock_codex = MagicMock()
        mock_codex.stdout = MagicMock()

        mock_jq = MagicMock()
        mock_jq.communicate.side_effect = subprocess.TimeoutExpired("jq", 60)

        mock_popen.side_effect = [mock_codex, mock_jq]

        cli = CodexCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.codex.subprocess.Popen')
    @patch('src.llm.codex.subprocess.run')
    def test_generate_exception(self, mock_run, mock_popen):
        """Test generate handles general exceptions"""
        from src.llm.codex import CodexCLI

        mock_run.return_value = MagicMock(returncode=0)
        mock_popen.side_effect = Exception("Unknown error")

        cli = CodexCLI()
        result = cli.generate("Hello")

        assert result is None

    @patch('src.llm.codex.subprocess.Popen')
    @patch('src.llm.codex.subprocess.run')
    def test_generate_raw_output(self, mock_run, mock_popen):
        """Test generate with raw output (non-JSON)"""
        from src.llm.codex import CodexCLI

        mock_run.return_value = MagicMock(returncode=0)

        mock_codex = MagicMock()
        mock_codex.stdout = MagicMock()

        mock_jq = MagicMock()
        mock_jq.communicate.return_value = ("Raw output", "")
        mock_jq.returncode = 0

        mock_popen.side_effect = [mock_codex, mock_jq]

        cli = CodexCLI()
        result = cli.generate("Hello")

        assert result == "Raw output"
