# examples/utils.py
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SQLExample:
    input: str
    query: str


class ExampleValidator:
    """Classe utilitaire pour valider les exemples SQL."""

    @staticmethod
    def validate_example(example: Dict) -> bool:
        """Valide un exemple SQL individuel."""
        return all(key in example for key in ['input', 'query'])

    @staticmethod
    def validate_examples_file(content: str) -> bool:
        """Valide le contenu complet du fichier d'exemples."""
        try:
            data = json.loads(content)
            return (
                    isinstance(data, dict) and
                    'examples' in data and
                    isinstance(data['examples'], list) and
                    all(isinstance(ex, dict) for ex in data['examples']) and
                    all(ExampleValidator.validate_example(ex) for ex in data['examples'])
            )
        except json.JSONDecodeError:
            return False
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False