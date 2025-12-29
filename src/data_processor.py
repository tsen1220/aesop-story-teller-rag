"""Data processing module: Process Aesop's Fables data"""
import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FableDataProcessor:
    """Process Aesop's Fables data"""
    def __init__(self):
        # Test data processing
        self.raw_data_path = os.getenv('RAW_DATA_PATH', 'data/aesop_fables_raw.json')
        self.processed_data_path = os.getenv('DATA_PATH', 'data/aesop_fables_processed.json')
        self.fables: List[Dict] = []

    def load_data(self) -> Dict:
        """Load raw JSON data"""
        self.load_raw_data()
        processed = self.process_fables()
        self.save_processed_data(processed)
        return self.get_statistics(processed)

    def load_raw_data(self) -> None:
        """Load raw JSON data"""
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.fables = data.get('stories', [])
        print(f"✓ Loaded {len(self.fables)} fables")

    def process_fables(self) -> List[Dict]:
        """Process fables and convert to vector database format"""
        processed_fables = []

        for fable in self.fables:
            # Merge story array into complete text
            story_text = ' '.join(fable.get('story', []))

            processed = {
                'id': f"fable_{fable.get('number', '00')}",
                'title': fable.get('title', ''),
                'content': story_text,
                'moral': fable.get('moral', ''),
                'language': 'en',
                'metadata': {
                    'number': fable.get('number', ''),
                    'characters': fable.get('characters', []),
                    'word_count': len(story_text.split())
                }
            }
            processed_fables.append(processed)

        print(f"✓ Processed {len(processed_fables)} fables")
        return processed_fables

    def save_processed_data(self, data: List[Dict]) -> None:
        """Save processed data"""
        output_file = Path(self.processed_data_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved data to {output_path}")

    def get_statistics(self, data: List[Dict]) -> Dict:
        """Get data statistics"""
        total_words = sum(item['metadata']['word_count'] for item in data)
        avg_words = total_words / len(data) if data else 0

        return {
            'total_fables': len(data),
            'total_words': total_words,
            'average_words_per_fable': round(avg_words, 2)
        }


if __name__ == '__main__':
    processor = FableDataProcessor()
    stats = processor.load_data()
    
    print(f"\nStatistics:")
    print(f"  Total fables: {stats['total_fables']}")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Average words per fable: {stats['average_words_per_fable']}")
