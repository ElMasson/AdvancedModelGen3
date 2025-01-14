# storage/utils.py
import os
from typing import List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_file_list(directory: str, extension: str = None) -> List[str]:
    """
    Obtient la liste des fichiers dans un répertoire.

    Args:
        directory: Répertoire à scanner
        extension: Extension de fichier optionnelle

    Returns:
        Liste des chemins de fichiers
    """
    try:
        files = []
        for filename in os.listdir(directory):
            if extension is None or filename.endswith(extension):
                files.append(os.path.join(directory, filename))
        return sorted(files)
    except Exception as e:
        logger.error(f"Error getting file list: {str(e)}")
        raise


def create_timestamped_filename(prefix: str, extension: str) -> str:
    """
    Crée un nom de fichier avec timestamp.

    Args:
        prefix: Préfixe du fichier
        extension: Extension du fichier

    Returns:
        Nom de fichier avec timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def ensure_directory(directory: str):
    """
    S'assure qu'un répertoire existe.

    Args:
        directory: Chemin du répertoire
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise