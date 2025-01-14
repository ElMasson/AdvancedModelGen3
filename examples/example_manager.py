# examples/example_manager.py
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SQLExample:
    input: str
    query: str


class ExampleManager:
    """Gestionnaire des exemples SQL."""

    def __init__(self):
        self.examples: List[SQLExample] = []
        self.last_update: Optional[datetime] = None
        self.status: Dict = {
            'loaded': False,
            'count': 0,
            'error': None
        }

    def load_examples(self, content: str) -> bool:
        """Charge les exemples à partir du contenu JSON."""
        try:
            data = json.loads(content)

            # Validation du format
            if not isinstance(data, dict) or 'examples' not in data:
                self.status = {
                    'loaded': False,
                    'count': 0,
                    'error': 'Invalid JSON format: missing examples key'
                }
                return False

            examples = data['examples']
            if not isinstance(examples, list):
                self.status = {
                    'loaded': False,
                    'count': 0,
                    'error': 'Invalid JSON format: examples must be a list'
                }
                return False

            # Chargement des exemples
            self.examples = [
                SQLExample(input=ex['input'], query=ex['query'])
                for ex in examples
                if 'input' in ex and 'query' in ex
            ]

            self.last_update = datetime.now()
            self.status = {
                'loaded': True,
                'count': len(self.examples),
                'error': None
            }

            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            self.status = {
                'loaded': False,
                'count': 0,
                'error': f"Invalid JSON: {str(e)}"
            }
            return False

        except Exception as e:
            logger.error(f"Error loading examples: {str(e)}")
            self.status = {
                'loaded': False,
                'count': 0,
                'error': str(e)
            }
            return False

    def get_example_catalog(self) -> List[Dict]:
        """Retourne le catalogue des exemples."""
        return [
            {
                'input': example.input,
                'query': example.query
            }
            for example in self.examples
        ]

    def get_status(self) -> Dict:
        """Retourne le statut actuel."""
        return {
            'loaded': self.status['loaded'],
            'count': self.status['count'],
            'error': self.status['error'],
            'last_update': self.last_update.isoformat() if self.last_update else None
        }