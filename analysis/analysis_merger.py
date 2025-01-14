# analysis/analysis_merger.py

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class AnalysisMerger:
    """Classe responsable de la fusion des analyses SQL et schéma."""

    @staticmethod
    def merge_analyses(schema_analysis: Dict[str, Any], sql_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fusionne les analyses de schéma et SQL en un catalogue complet.

        Args:
            schema_analysis: Résultats de l'analyse du schéma
            sql_analysis: Résultats de l'analyse SQL (optionnel)

        Returns:
            Dict contenant l'analyse fusionnée
        """
        try:
            if not sql_analysis:
                return schema_analysis

            # Copie profonde pour ne pas modifier les originaux
            merged = copy.deepcopy(schema_analysis)

            # Pour chaque schéma
            for schema_name, schema_data in merged.items():
                if not isinstance(schema_data, dict):
                    continue

                tables = schema_data.get('tables', {})
                if not isinstance(tables, dict):
                    continue

                # Pour chaque table du schéma
                for table_name, table_data in tables.items():
                    # Recherche des règles SQL correspondantes
                    sql_rules = AnalysisMerger._find_sql_rules(
                        sql_analysis,
                        schema_name,
                        table_name
                    )

                    if sql_rules:
                        # Mise à jour ou création de llm_analysis
                        if isinstance(table_data, dict):
                            if 'llm_analysis' not in table_data:
                                table_data['llm_analysis'] = {}

                            # Fusion des règles existantes avec les nouvelles
                            existing_rules = table_data['llm_analysis'].get('sql_rules', {})
                            table_data['llm_analysis']['sql_rules'] = AnalysisMerger._merge_rules(
                                existing_rules,
                                sql_rules
                            )

                            # Pour chaque colonne, ajout des règles SQL spécifiques
                            columns = table_data.get('columns', {})
                            if isinstance(columns, dict):
                                for col_name, col_data in columns.items():
                                    col_rules = AnalysisMerger._find_column_rules(
                                        sql_analysis,
                                        schema_name,
                                        table_name,
                                        col_name
                                    )
                                    if col_rules and isinstance(col_data, dict):
                                        if 'llm_analysis' not in col_data:
                                            col_data['llm_analysis'] = {}
                                        col_data['llm_analysis']['sql_rules'] = col_rules

            return merged

        except Exception as e:
            logger.error(f"Error merging analyses: {str(e)}")
            return schema_analysis

    @staticmethod
    def _find_sql_rules(
            sql_analysis: Dict[str, Any],
            schema_name: str,
            table_name: str
    ) -> Dict[str, Any]:
        """
        Trouve les règles SQL pour une table spécifique.

        Args:
            sql_analysis: Analyse SQL complète
            schema_name: Nom du schéma
            table_name: Nom de la table

        Returns:
            Dict des règles trouvées
        """
        try:
            qualified_name = f"{schema_name}.{table_name}"
            simple_name = table_name

            result = {
                'table_rules': [],
                'column_rules': {},
                'joins': []
            }

            # Recherche des règles de table avec vérification détaillée
            for rule in sql_analysis.get('table_rules', []):
                tables = rule.get('tables', [])
                if isinstance(tables, list):
                    if any(table in [qualified_name, simple_name] for table in tables):
                        result['table_rules'].append(rule)
                    # Recherche dans la description et le contexte
                    elif any(name in rule.get('description', '').lower()
                             for name in [qualified_name.lower(), simple_name.lower()]):
                        result['table_rules'].append(rule)

            # Recherche des règles de colonnes
            column_rules = sql_analysis.get('column_rules', {})
            for col_name, rules in column_rules.items():
                # Gestion des noms qualifiés
                if '.' in col_name:
                    rule_schema, rule_table, rule_col = col_name.split('.')
                    if rule_schema == schema_name and rule_table == table_name:
                        result['column_rules'][rule_col] = rules
                else:
                    # Vérification des règles non qualifiées
                    for rule in rules:
                        if any(table in [qualified_name, simple_name]
                               for table in rule.get('tables', [])):
                            if col_name not in result['column_rules']:
                                result['column_rules'][col_name] = []
                            result['column_rules'][col_name].append(rule)

            # Recherche des jointures avec vérification étendue
            for join in sql_analysis.get('joins', []):
                source = join.get('source_table', '')
                target = join.get('target_table', '')
                if any(table in [source, target]
                       for table in [qualified_name, simple_name]):
                    result['joins'].append(join)

            return result

        except Exception as e:
            logger.error(f"Error finding SQL rules: {str(e)}")
            return {'table_rules': [], 'column_rules': {}, 'joins': []}

    @staticmethod
    def _find_column_rules(
            sql_analysis: Dict[str, Any],
            schema_name: str,
            table_name: str,
            column_name: str
    ) -> List[Dict[str, Any]]:
        """
        Trouve les règles SQL pour une colonne spécifique.

        Args:
            sql_analysis: Analyse SQL complète
            schema_name: Nom du schéma
            table_name: Nom de la table
            column_name: Nom de la colonne

        Returns:
            Liste des règles trouvées
        """
        try:
            rules = []
            qualified_col = f"{schema_name}.{table_name}.{column_name}"

            # Recherche dans les règles de colonnes
            column_rules = sql_analysis.get('column_rules', {})

            # Vérification des différents formats possibles
            for col, col_rules in column_rules.items():
                if col in [qualified_col, column_name]:
                    rules.extend(col_rules)
                elif '.' in col:
                    _, _, rule_col = col.split('.')
                    if rule_col == column_name:
                        # Vérifie si la règle concerne la bonne table
                        for rule in col_rules:
                            if any(table in [f"{schema_name}.{table_name}", table_name]
                                   for table in rule.get('tables', [])):
                                rules.append(rule)

            return rules

        except Exception as e:
            logger.error(f"Error finding column rules: {str(e)}")
            return []

    @staticmethod
    def _merge_rules(
            existing_rules: Dict[str, Any],
            new_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fusionne les règles existantes avec les nouvelles.

        Args:
            existing_rules: Règles existantes
            new_rules: Nouvelles règles

        Returns:
            Dict des règles fusionnées
        """
        try:
            merged = {
                'table_rules': existing_rules.get('table_rules', []) + new_rules.get('table_rules', []),
                'column_rules': {},
                'joins': existing_rules.get('joins', []) + new_rules.get('joins', [])
            }

            # Fusion des règles de colonnes
            all_columns = set(existing_rules.get('column_rules', {}).keys()) | \
                          set(new_rules.get('column_rules', {}).keys())

            for col in all_columns:
                merged['column_rules'][col] = (
                        existing_rules.get('column_rules', {}).get(col, []) +
                        new_rules.get('column_rules', {}).get(col, [])
                )

            return merged

        except Exception as e:
            logger.error(f"Error merging rules: {str(e)}")
            return {'table_rules': [], 'column_rules': {}, 'joins': []}