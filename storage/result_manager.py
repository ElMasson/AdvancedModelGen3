# storage/result_manager.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from core.config import Config
from core.utils import JSONEncoder
import shutil

logger = logging.getLogger(__name__)


class ResultManager:
    """Gestionnaire des résultats d'analyse."""

    def __init__(self):
        """Initialise le gestionnaire de résultats."""
        self.output_dir = Config.OUTPUT_DIR
        self.intermediate_dir = Config.INTERMEDIATE_OUTPUT_DIR
        self.cache_dir = os.path.join(self.output_dir, 'cache')
        self._ensure_directories()

    def _ensure_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        for directory in [self.output_dir, self.intermediate_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)

    def save_analysis_results(self, results: Dict[str, Any], prefix: str = "analysis") -> str:
        """
        Sauvegarde les résultats d'analyse complets.

        Args:
            results: Résultats à sauvegarder
            prefix: Préfixe pour le nom du fichier

        Returns:
            Chemin du fichier sauvegardé
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, cls=JSONEncoder, indent=2)

            logger.info(f"Analysis results saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise

    def save_llm_analysis(self,
                          schema: str,
                          table: str,
                          analysis_data: Dict,
                          cache: bool = True) -> str:
        """
        Sauvegarde l'analyse LLM avec option de cache.

        Args:
            schema: Nom du schéma
            table: Nom de la table
            analysis_data: Données d'analyse
            cache: Utiliser le cache

        Returns:
            Chemin du fichier sauvegardé
        """
        try:
            # Génération du nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_analysis_{schema}_{table}_{timestamp}.json"

            # Choix du répertoire selon le mode cache
            target_dir = self.cache_dir if cache else self.output_dir
            filepath = os.path.join(target_dir, filename)

            # Sauvegarde des données
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'schema': schema,
                        'table': table,
                        'timestamp': timestamp,
                        'cached': cache
                    },
                    'analysis': analysis_data
                }, f, cls=JSONEncoder, indent=2)

            logger.info(f"LLM analysis saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving LLM analysis: {str(e)}")
            raise

    def load_analysis_results(self, filepath: str) -> Dict[str, Any]:
        """
        Charge des résultats d'analyse.

        Args:
            filepath: Chemin du fichier

        Returns:
            Résultats d'analyse
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            raise

    def get_cached_analysis(self,
                            schema: str,
                            table: str,
                            max_age_hours: int = 24) -> Optional[Dict]:
        """
        Récupère une analyse en cache si disponible.

        Args:
            schema: Nom du schéma
            table: Nom de la table
            max_age_hours: Âge maximum du cache en heures

        Returns:
            Analyse en cache ou None
        """
        try:
            # Recherche des fichiers correspondants
            pattern = f"llm_analysis_{schema}_{table}_"
            matching_files = [
                f for f in os.listdir(self.cache_dir)
                if f.startswith(pattern) and f.endswith('.json')
            ]

            if not matching_files:
                return None

            # Tri par date de modification
            latest_file = max(
                matching_files,
                key=lambda f: os.path.getmtime(os.path.join(self.cache_dir, f))
            )

            # Vérification de l'âge
            file_path = os.path.join(self.cache_dir, latest_file)
            file_age_hours = (
                                     datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                             ).total_seconds() / 3600

            if file_age_hours > max_age_hours:
                return None

            # Chargement des données
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error getting cached analysis: {str(e)}")
            return None

    def cleanup_old_files(self, max_age_days: int = 7):
        """
        Nettoie les anciens fichiers.

        Args:
            max_age_days: Âge maximum des fichiers en jours
        """
        try:
            current_time = datetime.now()

            for directory in [self.output_dir, self.intermediate_dir, self.cache_dir]:
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)

                    if os.path.isfile(filepath):
                        file_age = current_time - datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        )

                        if file_age.days > max_age_days:
                            os.remove(filepath)
                            logger.info(f"Removed old file: {filepath}")

        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")

    def export_documentation(self,
                             analysis_results: Dict,
                             format: str = "markdown") -> str:
        """
        Exporte la documentation des analyses.

        Args:
            analysis_results: Résultats d'analyse
            format: Format d'export

        Returns:
            Chemin du fichier exporté
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"documentation_{timestamp}.{format}"
            filepath = os.path.join(self.output_dir, filename)

            if format == "markdown":
                content = self._generate_markdown_doc(analysis_results)
            else:
                raise ValueError(f"Unsupported format: {format}")

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Documentation exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting documentation: {str(e)}")
            raise

    def _generate_markdown_doc(self, analysis_results: Dict) -> str:
        """
        Génère la documentation au format Markdown.

        Args:
            analysis_results: Résultats d'analyse

        Returns:
            Documentation au format Markdown
        """
        doc = ["# Database Analysis Documentation\n"]
        doc.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Métadonnées globales
        if 'metadata' in analysis_results:
            doc.append("## Analysis Overview\n")
            for key, value in analysis_results['metadata'].items():
                doc.append(f"- **{key}**: {value}\n")

        # Analyses par objet
        for obj_type in ['tables', 'views']:
            if obj_type in analysis_results:
                doc.append(f"\n## {obj_type.title()}\n")

                for obj_name, obj_data in analysis_results[obj_type].items():
                    doc.append(f"\n### {obj_name}\n")

                    # Analyse LLM
                    if 'llm_analysis' in obj_data:
                        doc.append("#### AI Analysis\n")
                        doc.append(self._format_llm_analysis(obj_data['llm_analysis']))

                    # Colonnes
                    if 'columns' in obj_data:
                        doc.append("\n#### Columns\n")
                        for col in obj_data['columns']:
                            doc.append(f"- **{col['name']}** ({col['type']})")
                            if 'description' in col:
                                doc.append(f"  - {col['description']}")

        return "\n".join(doc)

    def _format_llm_analysis(self, llm_analysis: Dict) -> str:
        """
        Formate l'analyse LLM pour la documentation.

        Args:
            llm_analysis: Analyse LLM

        Returns:
            Texte formaté
        """
        sections = []

        for section, content in llm_analysis.items():
            if section != 'error':
                sections.append(f"##### {section.replace('_', ' ').title()}\n")

                if isinstance(content, dict):
                    for key, value in content.items():
                        sections.append(f"- **{key}**: {value}")
                elif isinstance(content, list):
                    for item in content:
                        sections.append(f"- {item}")
                else:
                    sections.append(str(content))

                sections.append("")

        return "\n".join(sections)