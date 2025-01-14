import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    DATABASE_CONNECTIONS_FILE = "connections.json"
    OUTPUT_DIR = "output"
    INTERMEDIATE_OUTPUT_DIR = "intermediate_output"
    # Configuration LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_ENABLED = bool(OPENAI_API_KEY)  # Active seulement si la clé est présente
    # Paramètres LLM
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # timeout en secondes
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Chemins de fichiers
    DATABASE_CONNECTIONS_FILE = "connections.json"
    OUTPUT_DIR = "output"
    INTERMEDIATE_OUTPUT_DIR = "intermediate_output"

    # Nouveaux paramètres SQL
    SQL_EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "sql_examples")
    DEFAULT_SQL_EXAMPLE_FILE = "sql_example.json"

    # Patterns SQL pour l'analyse
    SQL_PATTERNS = {
        "aggregation": ["COUNT", "SUM", "AVG", "MIN", "MAX"],
        "grouping": ["GROUP BY"],
        "filtering": ["WHERE", "HAVING"],
        "joining": ["JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN"]
    }

    # Paramètres de validation SQL
    SQL_VALIDATION = {
        "max_file_size": 5 * 1024 * 1024,  # 5MB
        "required_fields": ["input", "query"],
        "allowed_query_types": ["SELECT", "INSERT", "UPDATE", "DELETE"]
    }

    @classmethod
    def check_llm_configuration(cls):
        """Vérifie et log le statut de la configuration LLM."""
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY non définie. L'analyse LLM sera désactivée.")
            return False
        return True

    @classmethod
    def ensure_directories(cls) -> None:
        """S'assure que tous les répertoires nécessaires existent."""
        directories = [
            cls.OUTPUT_DIR,
            cls.INTERMEDIATE_OUTPUT_DIR,
            cls.SQL_EXAMPLES_DIR
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    @classmethod
    def get_sql_examples_path(cls, filename: str = None) -> str:
        """
        Obtient le chemin complet pour un fichier d'exemples SQL.

        Args:
            filename: Nom du fichier optionnel

        Returns:
            Chemin complet du fichier
        """
        if filename is None:
            filename = cls.DEFAULT_SQL_EXAMPLE_FILE
        return os.path.join(cls.SQL_EXAMPLES_DIR, filename)


# Création automatique des répertoires au chargement
Config.ensure_directories()

