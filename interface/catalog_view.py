# interface/catalog_view.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import numpy as np
from core.config import Config

logger = logging.getLogger(__name__)


class CatalogView:
    """Vue du catalogue de données fusionné."""

    def render_catalog(self, catalog_data: Dict[str, Any]):
        """
        Affiche le catalogue complet avec données fusionnées.
        Utilise exactement le même format que Data Model Analysis.

        Args:
            catalog_data: Données du catalogue fusionné
        """
        try:
            schemas = list(catalog_data.keys())
            if not schemas:
                st.info("No schemas available in the catalog.")
                return

            # Sélection du schéma
            selected_schema = st.selectbox(
                "Select Schema", schemas, key="catalog_schema_select"
            )

            if selected_schema:
                schema_results = catalog_data[selected_schema]

                # Métriques globales
                self._render_schema_metrics(schema_results)

                # Navigation par onglets comme dans ResultsView
                tabs = st.tabs(["Overview", "Tables", "Views"])

                with tabs[0]:
                    self._render_overview_tab(schema_results)

                with tabs[1]:
                    self._render_tables_tab(schema_results)

                with tabs[2]:
                    self._render_views_tab(schema_results)

        except Exception as e:
            logger.error(f"Error rendering catalog: {str(e)}")
            st.error("Error displaying catalog data")

    def _render_schema_metrics(self, schema_results: Dict[str, Any]):
        """Affiche les métriques du schéma."""
        try:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Tables",
                    len(schema_results.get('tables', {}))
                )

            with col2:
                st.metric(
                    "Views",
                    len(schema_results.get('views', {}))
                )

            with col3:
                total_rules = sum(
                    len(table.get('llm_analysis', {}).get('sql_rules', {}).get('table_rules', []))
                    for table in schema_results.get('tables', {}).values()
                )
                st.metric("Business Rules", total_rules)

            with col4:
                total_cols = sum(
                    len(table.get('columns', {}))
                    for table in schema_results.get('tables', {}).values()
                )
                st.metric("Total Columns", total_cols)

        except Exception as e:
            logger.error(f"Error rendering schema metrics: {str(e)}")
            st.error("Error displaying metrics")

    def _create_statistics_dataframe(self, tables_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Crée un DataFrame avec les statistiques des tables.

        Args:
            tables_data: Données des tables

        Returns:
            DataFrame avec les statistiques
        """
        try:
            stats_data = []
            for table_name, table_data in tables_data.items():
                row = {
                    'Table Name': table_name,
                    'Columns': len(table_data.get('columns', {})),
                    'Business Rules': len(table_data.get('llm_analysis', {})
                                          .get('business_rules', [])),
                    'SQL Rules': len(table_data.get('llm_analysis', {})
                                     .get('sql_rules', {}).get('table_rules', [])),
                }
                stats_data.append(row)

            return pd.DataFrame(stats_data)

        except Exception as e:
            logger.error(f"Error creating statistics DataFrame: {str(e)}")
            return pd.DataFrame()

    def _render_size_distribution(self, tables_data: Dict[str, Any]):
        """
        Affiche la distribution des tailles des tables.

        Args:
            tables_data: Données des tables
        """
        try:
            sizes_data = {
                table_name: len(table_data.get('columns', {}))
                for table_name, table_data in tables_data.items()
            }

            fig = px.bar(
                x=list(sizes_data.keys()),
                y=list(sizes_data.values()),
                title="Table Sizes (Number of Columns)",
                labels={'x': 'Table', 'y': 'Number of Columns'}
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error rendering size distribution: {str(e)}")
            st.error("Error displaying size distribution")

    def _render_overview_tab(self, schema_results: Dict[str, Any]):
        """Affiche l'onglet Overview."""
        try:
            st.header("Schema Overview")

            # Statistiques de tables
            if 'tables' in schema_results:
                st.subheader("Tables Statistics")
                stats_df = self._create_statistics_dataframe(schema_results['tables'])
                st.dataframe(stats_df, use_container_width=True)

                # Graphique de distribution
                st.subheader("Table Sizes Distribution")
                self._render_size_distribution(schema_results['tables'])

            # Résumé des règles métier
            self._render_business_rules_summary(schema_results)

        except Exception as e:
            logger.error(f"Error rendering overview tab: {str(e)}")
            st.error("Error displaying overview")

    def _render_tables_tab(self, schema_results: Dict[str, Any]):
        """Affiche l'onglet Tables avec les règles fusionnées."""
        try:
            st.header("Tables Analysis")

            for table_name, table_data in schema_results.get('tables', {}).items():
                with st.expander(f"📊 {table_name}"):
                    # Table Description
                    st.markdown("### Description")
                    st.markdown(table_data.get('description', 'No description available'))

                    # Business Rules
                    st.markdown("### Business Rules")
                    llm_analysis = table_data.get('llm_analysis', {})

                    # Schema Rules
                    if 'business_rules' in llm_analysis:
                        st.markdown("#### Schema Analysis Rules")
                        for rule in llm_analysis['business_rules']:
                            st.markdown(f"- {rule}")

                    # SQL Rules
                    sql_rules = llm_analysis.get('sql_rules', {})
                    if sql_rules.get('table_rules'):
                        st.markdown("#### SQL Analysis Rules")
                        for rule in sql_rules['table_rules']:
                            with st.expander(f"📋 {rule['description']}"):
                                st.markdown(f"**Impact:** {rule.get('impact', 'N/A')}")
                                if 'business_justification' in rule:
                                    st.markdown(f"**Justification:** {rule['business_justification']}")
                                if rule.get('source_queries'):
                                    st.markdown("**Reference Queries:**")
                                    for query in rule['source_queries']:
                                        st.code(query, language='sql')

                    # Columns
                    st.markdown("### Columns")
                    for col_name, col_data in table_data.get('columns', {}).items():
                        with st.expander(f"📋 {col_name}"):
                            st.markdown(f"**Type:** `{col_data.get('type', 'N/A')}`")
                            st.markdown(f"**Description:** {col_data.get('description', 'N/A')}")

                            # Column SQL Rules
                            if col_name in sql_rules.get('column_rules', {}):
                                st.markdown("#### Business Rules")
                                for rule in sql_rules['column_rules'][col_name]:
                                    st.markdown(f"- {rule['description']}")
                                    if 'context' in rule:
                                        st.markdown(f"  *Context:* {rule['context']}")
                                    if rule.get('source_queries'):
                                        with st.expander("🔍 Reference Queries"):
                                            for query in rule['source_queries']:
                                                st.code(query, language='sql')

        except Exception as e:
            logger.error(f"Error rendering tables tab: {str(e)}")
            st.error("Error displaying tables")

    def _render_views_tab(self, schema_results: Dict[str, Any]):
        """Affiche l'onglet Views."""
        try:
            st.header("Views Analysis")
            views_data = schema_results.get('views', {})

            if not views_data:
                st.info("No views available in this schema.")
                return

            for view_name, view_data in views_data.items():
                with st.expander(f"👁️ {view_name}"):
                    st.markdown("### Description")
                    st.markdown(view_data.get('description', 'No description available'))

                    # Definition
                    if 'definition' in view_data:
                        st.markdown("### Definition")
                        st.code(view_data['definition'], language='sql')

                    # Business Rules
                    llm_analysis = view_data.get('llm_analysis', {})
                    if llm_analysis:
                        st.markdown("### Business Rules")
                        for rule in llm_analysis.get('business_rules', []):
                            st.markdown(f"- {rule}")

                    # Columns
                    if 'columns' in view_data:
                        st.markdown("### Columns")
                        for col_name, col_data in view_data['columns'].items():
                            with st.expander(f"📋 {col_name}"):
                                st.markdown(f"**Type:** `{col_data.get('type', 'N/A')}`")
                                st.markdown(f"**Description:** {col_data.get('description', 'N/A')}")

        except Exception as e:
            logger.error(f"Error rendering views tab: {str(e)}")
            st.error("Error displaying views")

    def _render_business_rules_summary(self, schema_results: Dict[str, Any]):
        """Affiche un résumé des règles métier."""
        try:
            st.subheader("Business Rules Summary")

            # Collecte des statistiques sur les règles
            rule_stats = {
                'Schema Rules': 0,
                'SQL Rules': 0,
                'Column Rules': 0
            }

            for table_data in schema_results.get('tables', {}).values():
                llm_analysis = table_data.get('llm_analysis', {})
                rule_stats['Schema Rules'] += len(llm_analysis.get('business_rules', []))
                rule_stats['SQL Rules'] += len(llm_analysis.get('sql_rules', {})
                                               .get('table_rules', []))
                rule_stats['Column Rules'] += sum(
                    len(rules) for rules in llm_analysis.get('sql_rules', {})
                    .get('column_rules', {}).values()
                )

            # Affichage des statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Schema Rules", rule_stats['Schema Rules'])
            with col2:
                st.metric("SQL Rules", rule_stats['SQL Rules'])
            with col3:
                st.metric("Column Rules", rule_stats['Column Rules'])

        except Exception as e:
            logger.error(f"Error rendering business rules summary: {str(e)}")
            st.error("Error displaying business rules summary")