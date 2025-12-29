"""Unit tests for data_processor module"""

import pytest
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from src.data_processor import FableDataProcessor


class TestFableDataProcessor:
    """Test FableDataProcessor class"""

    def test_init_default_paths(self):
        """Test initialization with default paths"""
        # Act
        processor = FableDataProcessor()

        # Assert
        assert processor.raw_data_path == 'data/aesop_fables_raw.json'
        assert processor.processed_data_path == 'data/aesop_fables_processed.json'
        assert processor.fables == []

    @patch.dict('os.environ', {
        'RAW_DATA_PATH': 'custom/raw.json',
        'DATA_PATH': 'custom/processed.json'
    })
    def test_init_custom_paths(self):
        """Test initialization with custom paths from environment"""
        # Act
        processor = FableDataProcessor()

        # Assert
        assert processor.raw_data_path == 'custom/raw.json'
        assert processor.processed_data_path == 'custom/processed.json'

    @patch('builtins.open', new_callable=mock_open, read_data='{"stories": [{"number": "01", "title": "Test"}]}')
    @patch('src.data_processor.json.load')
    def test_load_raw_data_success(self, mock_json_load, mock_file):
        """Test successfully loading raw data"""
        # Arrange
        sample_data = {
            "stories": [
                {
                    "number": "01",
                    "title": "Test Fable",
                    "story": ["Part one.", "Part two."],
                    "moral": "Test moral",
                    "characters": ["char1"]
                }
            ]
        }
        mock_json_load.return_value = sample_data
        processor = FableDataProcessor()

        # Act
        processor.load_raw_data()

        # Assert
        assert len(processor.fables) == 1
        assert processor.fables[0]['title'] == 'Test Fable'
        mock_file.assert_called_once_with(processor.raw_data_path, 'r', encoding='utf-8')

    @patch('builtins.open', side_effect=FileNotFoundError('File not found'))
    def test_load_raw_data_file_not_found(self, mock_file):
        """Test loading raw data when file doesn't exist"""
        # Arrange
        processor = FableDataProcessor()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            processor.load_raw_data()

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json{]')
    @patch('src.data_processor.json.load', side_effect=json.JSONDecodeError('error', 'doc', 0))
    def test_load_raw_data_invalid_json(self, mock_json_load, mock_file):
        """Test loading raw data with invalid JSON"""
        # Arrange
        processor = FableDataProcessor()

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            processor.load_raw_data()

    def test_process_fables_normal(self):
        """Test normal fable processing"""
        # Arrange
        processor = FableDataProcessor()
        processor.fables = [
            {
                "number": "01",
                "title": "Test Fable",
                "story": ["Part one.", "Part two."],
                "moral": "Test moral",
                "characters": ["char1", "char2"]
            }
        ]

        # Act
        result = processor.process_fables()

        # Assert
        assert len(result) == 1
        assert result[0]['id'] == 'fable_01'
        assert result[0]['title'] == 'Test Fable'
        assert result[0]['content'] == 'Part one. Part two.'
        assert result[0]['moral'] == 'Test moral'
        assert result[0]['language'] == 'en'
        assert result[0]['metadata']['number'] == '01'
        assert result[0]['metadata']['characters'] == ['char1', 'char2']
        assert result[0]['metadata']['word_count'] == 4  # "Part one. Part two." = 4 words

    def test_process_fables_empty_list(self):
        """Test processing empty fables list (boundary condition)"""
        # Arrange
        processor = FableDataProcessor()
        processor.fables = []

        # Act
        result = processor.process_fables()

        # Assert
        assert result == []

    def test_process_fables_missing_fields(self):
        """Test processing fables with missing fields"""
        # Arrange
        processor = FableDataProcessor()
        processor.fables = [
            {
                # Missing title, moral, characters
                "number": "01",
                "story": ["Test story."]
            }
        ]

        # Act
        result = processor.process_fables()

        # Assert
        assert len(result) == 1
        assert result[0]['id'] == 'fable_01'
        assert result[0]['title'] == ''
        assert result[0]['content'] == 'Test story.'
        assert result[0]['moral'] == ''
        assert result[0]['metadata']['characters'] == []

    @patch('src.data_processor.Path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.data_processor.json.dump')
    def test_save_processed_data_bug(self, mock_json_dump, mock_file, mock_path):
        """
        Test save_processed_data - documents known bug on line 67

        BUG: Line 67 uses undefined variable 'output_path' instead of 'output_file'
        This test documents the bug without fixing it as per requirements.
        """
        # Arrange
        processor = FableDataProcessor()
        test_data = [{"id": "fable_01", "title": "Test"}]

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path.return_value = mock_path_instance

        # Act & Assert - This will raise NameError due to the bug
        with pytest.raises(NameError, match="name 'output_path' is not defined"):
            processor.save_processed_data(test_data)

    @patch('src.data_processor.Path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.data_processor.json.dump')
    def test_save_processed_data_creates_directory(self, mock_json_dump, mock_file, mock_path):
        """
        Test that save_processed_data creates directory if it doesn't exist

        Note: This test will trigger the bug on line 67 where 'output_path' is undefined,
        so we expect a NameError. The bug is documented but not fixed per requirements.
        """
        # Arrange
        processor = FableDataProcessor()
        test_data = [{"id": "fable_01", "title": "Test"}]

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path.return_value = mock_path_instance

        # Act & Assert - will raise NameError due to bug
        with pytest.raises(NameError, match="name 'output_path' is not defined"):
            processor.save_processed_data(test_data)

    def test_get_statistics_normal(self):
        """Test getting statistics from normal data"""
        # Arrange
        processor = FableDataProcessor()
        test_data = [
            {
                "id": "fable_01",
                "metadata": {"word_count": 100}
            },
            {
                "id": "fable_02",
                "metadata": {"word_count": 200}
            },
            {
                "id": "fable_03",
                "metadata": {"word_count": 150}
            }
        ]

        # Act
        stats = processor.get_statistics(test_data)

        # Assert
        assert stats['total_fables'] == 3
        assert stats['total_words'] == 450
        assert stats['average_words_per_fable'] == 150.0

    def test_get_statistics_empty_data(self):
        """Test getting statistics from empty data (boundary condition)"""
        # Arrange
        processor = FableDataProcessor()
        test_data = []

        # Act
        stats = processor.get_statistics(test_data)

        # Assert
        assert stats['total_fables'] == 0
        assert stats['total_words'] == 0
        assert stats['average_words_per_fable'] == 0
