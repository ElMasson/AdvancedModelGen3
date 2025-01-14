import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from analysis.sql_rules_analyzer import SQLRulesAnalyzer, BusinessRule, JoinInfo

logger = logging.getLogger(__name__)


class SQLRulesView:
    """Composant de visualisation des règles métier SQL."""

    def __init__(self):
        """Initialise la vue des règles SQL."""
        self.analyzer = SQLRulesAnalyzer()

    def render_rules_tab(self, analysis_results: Optional[Dict[str, Any]] = None):
        """Affiche les résultats d'analyse des règles métier."""
        if not analysis_results:
            st.info("No SQL analysis results available. Please upload and analyze SQL examples first.")
            return

        # Sélection du schéma
        available_schemas = self._get_unique_schemas(analysis_results)
        selected_schema = st.selectbox(
            "Select Schema",
            available_schemas,
            key="schema_selector"
        )

        if selected_schema:
            # Sélection de la table
            available_tables = self._get_tables_for_schema(analysis_results, selected_schema)
            selected_table = st.selectbox(
                "Select Table",
                available_tables,
                key="table_selector"
            )

            if selected_table:
                # Section Table
                st.markdown(f"## Table: {selected_schema}.{selected_table}")

                # Règles métier niveau table
                business_rules = self._get_business_rules(analysis_results, selected_schema, selected_table)
                if business_rules:
                    for rule in business_rules:
                        st.markdown(f"📋 {rule['description']}")
                        with st.expander("Details"):
                            st.markdown(f"**Impact:** {rule.get('impact', 'N/A')}")
                            st.markdown(f"**Justification:** {rule.get('business_justification', 'N/A')}")

                            if rule.get('source_queries'):
                                st.markdown("**Reference Queries:**")
                                for query in rule['source_queries']:
                                    st.code(query, language='sql')
                else:
                    st.info("No specific business rules found for this table")

                # Relations au niveau table
                joins = self._get_joins_for_table(analysis_results, selected_schema, selected_table)
                if joins:
                    st.markdown("### Table Relations")
                    for join in joins:
                        st.markdown(f"🔗 {join['business_meaning']}")
                        with st.expander("Details"):
                            st.markdown(f"**From:** {join['source_table']}")
                            st.markdown(f"**To:** {join['target_table']}")
                            st.markdown(f"**Importance:** {join.get('importance', 'N/A')}")
                            st.markdown(f"**Justification:** {join.get('justification', 'N/A')}")

                            if join.get('source_queries'):
                                st.markdown("**Reference Queries:**")
                                for query in join['source_queries']:
                                    st.code(query, language='sql')

                # Section Colonnes
                st.markdown("## Columns")

                # Récupération de toutes les colonnes uniques mentionnées dans les règles
                column_rules = self._get_column_rules(analysis_results, selected_schema, selected_table)
                all_columns = self._get_all_columns(analysis_results, selected_schema, selected_table)

                # Affichage de chaque colonne dans un expander
                for column in all_columns:
                    with st.expander(f"📊 {column}"):
                        rules = column_rules.get(column, [])
                        if rules:
                            for rule in rules:
                                st.markdown("### Business Rules")
                                st.markdown(f"📋 {rule['description']}")

                                if rule.get('context'):
                                    st.markdown(f"**Context:** {rule['context']}")

                                # Relations avec d'autres colonnes
                                if rule.get('related_columns'):
                                    st.markdown("### Related Columns")
                                    for rel_col in rule['related_columns']:
                                        st.markdown(f"🔗 {rel_col}")

                                # Requêtes de référence
                                if rule.get('source_queries'):
                                    with st.expander("Reference Queries"):
                                        for query in rule['source_queries']:
                                            st.code(query, language='sql')
                        else:
                            st.info("No specific rules found for this column")

    def _get_all_columns(self, analysis_results: Dict[str, Any], schema: str, table_name: str) -> List[str]:
        """
        Récupère toutes les colonnes uniques pour une table.

        Args:
            analysis_results: Résultats d'analyse
            schema: Schéma sélectionné
            table_name: Nom de la table

        Returns:
            Liste des colonnes uniques
        """
        columns = set()
        qualified_table = f"{schema}.{table_name}"

        # Colonnes des règles
        for col_name in analysis_results.get('column_rules', {}):
            if '.' in col_name:
                rule_schema, rule_table, rule_col = col_name.split('.')
                if rule_schema == schema and rule_table == table_name:
                    columns.add(rule_col)
            else:
                # Vérifier si la colonne est liée à la table
                for rule in analysis_results['column_rules'][col_name]:
                    if qualified_table in rule.get('tables', []) or table_name in rule.get('tables', []):
                        columns.add(col_name)

        # Colonnes mentionnées dans les jointures
        for join in analysis_results.get('joins', []):
            if join.get('source_table') == qualified_table and join.get('source_column'):
                columns.add(join['source_column'])
            if join.get('target_table') == qualified_table and join.get('target_column'):
                columns.add(join['target_column'])

        return sorted(list(columns))

    def _get_unique_schemas(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extrait la liste des schémas uniques."""
        schemas = set()
        for rule in analysis_results.get('table_rules', []):
            for table in rule.get('tables', []):
                if '.' in table:
                    schemas.add(table.split('.')[0])
        return sorted(list(schemas))

    def _get_tables_for_schema(self, analysis_results: Dict[str, Any], schema: str) -> List[str]:
        """Extrait les tables pour un schéma donné."""
        tables = set()
        for rule in analysis_results.get('table_rules', []):
            for table in rule.get('tables', []):
                if table.startswith(f"{schema}."):
                    tables.add(table.split('.')[-1])
        return sorted(list(tables))

    def _render_table_analysis(self, analysis_results: Dict[str, Any], table_name: str):
        """Affiche l'analyse d'une table spécifique."""
        # Analyse globale
        st.markdown("## Analyse Globale")

        # Catégories
        st.markdown("### Catégories")
        categs = self._get_categories_for_table(analysis_results, table_name)
        for categ in categs:
            st.markdown(f"- {categ}")

        # Noms alternatifs
        st.markdown("### Noms alternatifs")
        alt_names = self._get_alternative_names(analysis_results, table_name)
        for name in alt_names:
            st.markdown(f"- {name}")

        # Objectif fonctionnel
        st.markdown("### Objectif fonctionnel")
        business_rules = self._get_business_rules(analysis_results, table_name)
        for rule in business_rules:
            st.markdown(f"- {rule['description']}")
            with st.expander("🔍 Détails"):
                st.markdown(f"**Impact:** {rule.get('impact', 'N/A')}")
                st.markdown(f"**Justification:** {rule.get('business_justification', 'N/A')}")
                if rule.get('source_queries'):
                    st.markdown("**Source Queries:**")
                    for query in rule['source_queries']:
                        st.code(query, language='sql')

        # Colonnes
        st.markdown("## Colonnes")
        column_rules = self._get_column_rules(analysis_results, table_name)

        for column, rules in column_rules.items():
            with st.expander(f"📊 {column}"):
                if rules:
                    for rule in rules:
                        st.markdown(f"**📋 Règle:** {rule['description']}")
                        st.markdown(f"**🔄 Contexte:** {rule.get('context', 'N/A')}")

                        # Relations
                        if rule.get('related_columns'):
                            st.markdown("**🔗 Colonnes liées:**")
                            for rel_col in rule['related_columns']:
                                st.markdown(f"- {rel_col}")

                        # Source queries dans un sous-expander
                        if rule.get('source_queries'):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown("**💡 Requêtes sources:**")
                            for query in rule['source_queries']:
                                st.code(query, language='sql', line_numbers=True)
                else:
                    st.info("Aucune règle métier significative identifiée pour cette colonne")

        # Relations
        st.markdown("## Relations")
        joins = self._get_joins_for_table(analysis_results, table_name)
        for join in joins:
            with st.expander(f"🔗 {join['source_table']} ↔ {join['target_table']}"):
                st.markdown(f"**Signification métier:** {join['business_meaning']}")
                st.markdown(f"**Importance:** {join.get('importance', 'N/A')}")
                st.markdown(f"**Justification:** {join.get('justification', 'N/A')}")

                if join.get('source_queries'):
                    st.markdown("**Requêtes sources:**")
                    for query in join['source_queries']:
                        st.code(query, language='sql')

    def _get_categories_for_table(self, analysis_results: Dict[str, Any], table_name: str) -> List[str]:
        """Extrait les catégories pour une table."""
        categories = set()
        for rule in analysis_results.get('table_rules', []):
            if table_name in rule.get('tables', []):
                if 'category' in rule:
                    categories.add(rule['category'])
        return sorted(list(categories)) or ["Human Resources", "Employee Management"]

    def _get_alternative_names(self, analysis_results: Dict[str, Any], table_name: str) -> List[str]:
        """Extrait les noms alternatifs pour une table."""
        names = set()
        for rule in analysis_results.get('table_rules', []):
            if table_name in rule.get('tables', []):
                if 'alternative_names' in rule:
                    names.update(rule['alternative_names'])
        return sorted(list(names)) or ["Employee Attendance Record", "Work Hours Log"]

    def _get_business_rules(self, analysis_results: Dict[str, Any], schema: str, table_name: str) -> List[Dict]:
        """
        Extrait les règles métier pour une table spécifique.

        Args:
            analysis_results: Résultats d'analyse
            schema: Schéma sélectionné
            table_name: Nom de la table

        Returns:
            Liste des règles métier filtrées
        """
        qualified_table = f"{schema}.{table_name}"
        return [
            rule for rule in analysis_results.get('table_rules', [])
            if qualified_table in rule.get('tables', []) or table_name in rule.get('tables', [])
        ]

    def _get_column_rules(self, analysis_results: Dict[str, Any], schema: str, table_name: str) -> Dict[
        str, List[Dict]]:
        """
        Extrait les règles par colonne pour une table spécifique.

        Args:
            analysis_results: Résultats d'analyse
            schema: Schéma sélectionné
            table_name: Nom de la table

        Returns:
            Dictionnaire des règles par colonne
        """
        qualified_table = f"{schema}.{table_name}"
        column_rules = {}

        for col_name, rules in analysis_results.get('column_rules', {}).items():
            # Vérifier si la colonne appartient à la table sélectionnée
            if '.' in col_name:
                rule_schema, rule_table, rule_col = col_name.split('.')
                if rule_schema == schema and rule_table == table_name:
                    column_rules[rule_col] = rules
            else:
                # Cas où la colonne n'a pas le schéma/table qualifié
                for rule in rules:
                    if qualified_table in rule.get('tables', []) or table_name in rule.get('tables', []):
                        if col_name not in column_rules:
                            column_rules[col_name] = []
                        column_rules[col_name].append(rule)

        return column_rules

    def _get_joins_for_table(self, analysis_results: Dict[str, Any], schema: str, table_name: str) -> List[Dict]:
        """
        Extrait les jointures pour une table spécifique.

        Args:
            analysis_results: Résultats d'analyse
            schema: Schéma sélectionné
            table_name: Nom de la table

        Returns:
            Liste des jointures filtrées
        """
        qualified_table = f"{schema}.{table_name}"
        return [
            join for join in analysis_results.get('joins', [])
            if (qualified_table in [join['source_table'], join['target_table']] or
                table_name in [join['source_table'], join['target_table']])
        ]