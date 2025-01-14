# analysis/llm_analyzer.py

from openai import OpenAI
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
from core.config import Config
from core.utils import JSONEncoder

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Analyseur utilisant les LLMs pour l'analyse fonctionnelle et métier des structures de base de données.
    """

    def __init__(self):
        """Initialise l'analyseur LLM."""
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.default_model = "gpt-4o-mini"
        self.retry_attempts = 3
        self.retry_delay = 1  # secondes
        logger.info("LLM Analyzer initialized successfully")

    def analyze_database_object(self,
                                object_data: Dict,
                                object_type: str = "table") -> Dict:
        """
        Point d'entrée principal pour l'analyse d'un objet de base de données.
        Orchestre l'analyse complète incluant la structure et les colonnes.

        Args:
            object_data: Données de l'objet à analyser
            object_type: Type d'objet ('table' ou 'view')

        Returns:
            Dict contenant l'analyse complète
        """
        try:
            logger.info(f"Starting LLM analysis for {object_type}")

            # Analyse de base via analyze_table
            base_analysis = self.analyze_table(object_data)

            # Ajout de métadonnées supplémentaires
            enriched_analysis = {
                'object_type': object_type,
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.default_model,
                    'analysis_version': '1.0'
                },
                'analysis_results': base_analysis
            }

            logger.info(f"LLM analysis completed successfully for {object_type}")
            return enriched_analysis

        except Exception as e:
            error_msg = f"Error during database object analysis: {str(e)}"
            logger.error(error_msg)
            return {
                'object_type': object_type,
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error_message': str(e)
                },
                'analysis_results': self._get_error_analysis(str(e))
            }

    def analyze_with_prompt(self, prompt: str) -> Dict:
        """
        Analyse basée sur un prompt spécifique.

        Args:
            prompt: Le prompt à utiliser pour l'analyse

        Returns:
            Dict contenant les résultats de l'analyse
        """
        try:
            # Ajout d'instructions explicites pour le format JSON
            formatted_prompt = f"""
            Analyze the following and provide response in valid JSON format:
            {prompt}

            Respond ONLY with a valid JSON object containing:
            {{
                "analysis": {{
                    "description": "Your analysis description",
                    "findings": ["finding1", "finding2", ...],
                    "recommendations": ["rec1", "rec2", ...],
                    "confidence": 0.95
                }},
                "relationships": [
                    {{
                        "relation_key": "source__target",
                        "type": "relationship type",
                        "description": "relationship description",
                        "confidence": 0.8,
                        "recommendations": []
                    }}
                ]
            }}
            """

            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a database expert. Always respond with valid JSON."
                    },
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Nettoyage et validation du JSON
            try:
                # Tentative de parse direct
                cleaned_content = self._clean_json_string(content)
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                # Si échec, retourner une structure par défaut
                logger.warning(f"Failed to parse LLM response as JSON. Using default structure.")
                return {
                    "analysis": {
                        "description": "Analysis not available",
                        "findings": [],
                        "recommendations": [],
                        "confidence": 0.0
                    },
                    "relationships": []
                }

        except Exception as e:
            logger.error(f"Error in analyze_with_prompt: {str(e)}")
            return {
                "analysis": {
                    "description": f"Error during analysis: {str(e)}",
                    "findings": [],
                    "recommendations": [],
                    "confidence": 0.0
                },
                "relationships": []
            }

    def _clean_json_string(self, content: str) -> str:
        """
        Nettoie une chaîne pour en faire un JSON valide.

        Args:
            content: Contenu à nettoyer

        Returns:
            str: JSON valide
        """
        # Supprimer les backticks et identifiants de code
        content = content.replace('```json', '').replace('```', '')

        # Supprimer les espaces et sauts de ligne au début et à la fin
        content = content.strip()

        # Si le contenu ne commence pas par {, chercher le premier {
        if not content.startswith('{'):
            start_idx = content.find('{')
            if start_idx != -1:
                content = content[start_idx:]

        # Si le contenu ne finit pas par }, chercher le dernier }
        if not content.endswith('}'):
            end_idx = content.rfind('}')
            if end_idx != -1:
                content = content[:end_idx + 1]

        return content

    def analyze_table(self, table_data: Dict) -> Dict:
        """
        Analyse une table avec le LLM.

        Args:
            table_data: Données de la table à analyser

        Returns:
            Dict contenant l'analyse LLM
        """
        try:
            # Analyse globale de la table
            structure_analysis = self._analyze_object_structure(table_data)

            # Analyse des colonnes
            column_analyses = {}
            if 'columns' in table_data:
                for column in table_data['columns']:
                    column_analyses[column['name']] = self._analyze_column(
                        column,
                        table_context=table_data.get('metadata', {})
                    )

            return {
                'structure_analysis': structure_analysis,
                'column_analyses': column_analyses,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'model_used': self.default_model
                }
            }

        except Exception as e:
            logger.error(f"Error in analyze_table: {str(e)}")
            return self._get_error_analysis(str(e))

    def _analyze_object_structure(self, object_data: Dict) -> Dict:
        """
        Analyse la structure globale d'un objet.

        Args:
            object_data: Données de l'objet

        Returns:
            Dict contenant l'analyse de structure
        """
        try:
            prompt = f"""
            Analyze this database object and provide a complete business and functional analysis in French.

            Object Information:
            {json.dumps(object_data.get('metadata', {}), indent=2)}

            Columns:
            {json.dumps([{
                'name': col['name'],
                'type': col['type'],
                'description': col.get('description', 'No description available')
            } for col in object_data.get('columns', [])], indent=2)}

            Provide your analysis in this exact JSON format:
            {{
                "purpose": "Main business purpose and role",
                "categories": ["Category 1", "Category 2"],
                "alternative_names": ["Name 1", "Name 2"],
                "functional_description": "Detailed functional description",
                "use_cases": [
                    "Use case 1",
                    "Use case 2"
                ],
                "suggestions": [
                    "Suggestion 1",
                    "Suggestion 2"
                ],
                "detailed_analysis": {{
                    "summary": "Analysis summary",
                    "key_points": [
                        "Point 1",
                        "Point 2"
                    ],
                    "data_structure": {{
                        "description": "Structure description",
                        "relationships": ["Relationship 1", "Relationship 2"],
                        "integrity": "Integrity analysis",
                        "constraints": ["Constraint 1", "Constraint 2"]
                    }},
                    "usage_patterns": [
                        "Pattern 1",
                        "Pattern 2"
                    ]
                }}
            }}

            Provide specific, practical insights based on the provided schema written in French
            """

            # Appel au LLM avec retry
            for attempt in range(self.retry_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.default_model,
                        messages=[
                            {"role": "system", "content": "You are a database expert analyzing data structures."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2000
                    )

                    content = self._clean_json_string(response.choices[0].message.content)
                    analysis = json.loads(content)
                    return self._validate_structure_analysis(analysis)

                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    time.sleep(self.retry_delay * (attempt + 1))

        except Exception as e:
            logger.error(f"Error in _analyze_object_structure: {str(e)}")
            return self._get_error_structure_analysis(str(e))

    def _analyze_column(self, column_data: Dict, table_context: Dict) -> Dict:
        """
        Analyse une colonne avec le LLM.

        Args:
            column_data: Données de la colonne
            table_context: Contexte de la table

        Returns:
            Dict contenant l'analyse de la colonne
        """
        try:
            prompt = f"""
            Analyze this database column and provide a business and functional analysis in French
            .

            Column Information:
            {json.dumps(column_data, indent=2)}

            Table Context:
            {json.dumps(table_context, indent=2)}

            Provide your analysis in this exact JSON format:
            {{
                "business_purpose": "Main business purpose",
                "functional_description": "Detailed functional description",
                "data_characteristics": {{
                    "nature": "Nature of the data",
                    "expected_values": "Description of expected values",
                    "special_cases": ["Case 1", "Case 2"]
                }},
                "business_rules": [
                    "Rule 1",
                    "Rule 2"
                ],
                "data_quality": {{
                    "critical_aspects": ["Aspect 1", "Aspect 2"],
                    "validation_rules": ["Rule 1", "Rule 2"],
                    "recommendations": ["Recommendation 1", "Recommendation 2"]
                }},
                "relationships": {{
                    "dependencies": ["Dependency 1", "Dependency 2"],
                    "impact": ["Impact 1", "Impact 2"]
                }}
            }}
            """

            # Appel au LLM avec retry
            for attempt in range(self.retry_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.default_model,
                        messages=[
                            {"role": "system", "content": "You are a database expert analyzing columns."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000
                    )

                    content = self._clean_json_string(response.choices[0].message.content)
                    analysis = json.loads(content)
                    return self._validate_column_analysis(analysis)

                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    time.sleep(self.retry_delay * (attempt + 1))

        except Exception as e:
            logger.error(f"Error in _analyze_column: {str(e)}")
            return self._get_error_column_analysis(str(e))

    def _validate_structure_analysis(self, analysis: Dict) -> Dict:
        """
        Valide et normalise l'analyse de structure.

        Args:
            analysis: Analyse à valider

        Returns:
            Dict validé et normalisé
        """
        template = {
            "purpose": "Analysis not available",
            "categories": [],
            "alternative_names": [],
            "functional_description": "Analysis not available",
            "use_cases": [],
            "suggestions": [],
            "detailed_analysis": {
                "summary": "Analysis not available",
                "key_points": [],
                "data_structure": {
                    "description": "Analysis not available",
                    "relationships": [],
                    "integrity": "Analysis not available",
                    "constraints": []
                },
                "usage_patterns": []
            }
        }

        return self._ensure_structure(analysis, template)

    def _validate_column_analysis(self, analysis: Dict) -> Dict:
        """
        Valide et normalise l'analyse de colonne.

        Args:
            analysis: Analyse à valider

        Returns:
            Dict validé et normalisé
        """
        template = {
            "business_purpose": "Analysis not available",
            "functional_description": "Analysis not available",
            "data_characteristics": {
                "nature": "Analysis not available",
                "expected_values": "Analysis not available",
                "special_cases": []
            },
            "business_rules": [],
            "data_quality": {
                "critical_aspects": [],
                "validation_rules": [],
                "recommendations": []
            },
            "relationships": {
                "dependencies": [],
                "impact": []
            }
        }

        return self._ensure_structure(analysis, template)

    def _ensure_structure(self, data: Dict, template: Dict) -> Dict:
        """
        Assure qu'un dictionnaire suit une structure donnée.

        Args:
            data: Données à valider
            template: Structure attendue

        Returns:
            Dict validé et normalisé
        """
        result = {}

        for key, default_value in template.items():
            if key not in data:
                result[key] = default_value
            elif isinstance(default_value, dict):
                result[key] = self._ensure_structure(data[key], default_value)
            elif isinstance(default_value, list):
                result[key] = data[key] if isinstance(data[key], list) else default_value
            else:
                result[key] = data[key]

        return result

    def _clean_json_string(self, content: str) -> str:
        """
        Nettoie une chaîne JSON.

        Args:
            content: Contenu à nettoyer

        Returns:
            Chaîne JSON nettoyée
        """
        content = content.strip()
        content = content.replace('```json', '').replace('```', '')
        return content.strip()

    def _get_error_analysis(self, error_message: str) -> Dict:
        """
        Génère une analyse d'erreur.

        Args:
            error_message: Message d'erreur

        Returns:
            Dict d'erreur formaté
        """
        return {
            'error': error_message,
            'structure_analysis': self._get_error_structure_analysis(error_message),
            'column_analyses': {},
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
        }

    def _get_error_structure_analysis(self, error_message: str) -> Dict:
        """
        Génère une analyse de structure d'erreur.

        Args:
            error_message: Message d'erreur

        Returns:
            Dict d'erreur formaté
        """
        return {
            "error": error_message,
            "purpose": "Analysis failed",
            "categories": [],
            "alternative_names": [],
            "functional_description": "Analysis failed",
            "use_cases": [],
            "suggestions": [],
            "detailed_analysis": {
                "summary": "Analysis failed",
                "key_points": [],
                "data_structure": {
                    "description": "Analysis failed",
                    "relationships": [],
                    "integrity": "Analysis failed",
                    "constraints": []
                },
                "usage_patterns": []
            }
        }

    def _get_error_column_analysis(self, error_message: str) -> Dict:
        """
        Génère une analyse de colonne d'erreur.

        Args:
            error_message: Message d'erreur

        Returns:
            Dict d'erreur formaté
        """
        return {
            "error": error_message,
            "business_purpose": "Analysis failed",
            "functional_description": "Analysis failed",
            "data_characteristics": {
                "nature": "Analysis failed",
                "expected_values": "Analysis failed",
                "special_cases": []
            },
            "business_rules": [],
            "data_quality": {
                "critical_aspects": [],
                "validation_rules": [],
                "recommendations": []
            },
            "relationships": {
                "dependencies": [],
                "impact": []
            }
        }


def analyze_query(self, query: str, context: str = "") -> Dict:
    """
    Analyse une requête SQL avec le contexte.

    Args:
        query: Requête SQL à analyser
        context: Contexte de la requête

    Returns:
        Dict contenant l'analyse LLM
    """
    try:
        # Crée le prompt avec le contexte
        prompt = f"""
        Analyze this SQL query:

        Context: {context}
        Query: {query}

        Provide:
        1. A description of what the query does
        2. The business rules implemented
        3. Data relationships identified
        4. Any optimization suggestions
        
        Answer in French
        """

        # Appel de l'API LLM
        response = self._call_llm(prompt)

        # Parse et structure la réponse
        return {
            'description': response.get('description', ''),
            'business_rules': response.get('business_rules', []),
            'relationships': response.get('relationships', []),
            'suggestions': response.get('suggestions', [])
        }

    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}")
        return {}