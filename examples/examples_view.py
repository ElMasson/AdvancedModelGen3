# examples/examples_view.py
import streamlit as st
import json
from typing import Dict, Optional
from datetime import datetime
import logging
from .example_manager import ExampleManager
import pandas as pd
from analysis.sql_rules_analyzer import SQLRulesAnalyzer

logger = logging.getLogger(__name__)


# examples/examples_view.py
class ExamplesView:
    """Composant de visualisation des exemples SQL."""

    def __init__(self):
        """Initialise la vue des exemples."""
        if 'example_manager' not in st.session_state:
            st.session_state.example_manager = ExampleManager()
        self.example_manager = st.session_state.example_manager

    def render(self):
        """Affiche la section des exemples SQL."""
        # Section upload
        uploaded_file = st.file_uploader(
            "Upload Examples (JSON)",
            type=['json'],
            help="Upload a JSON file containing SQL examples with input/query pairs"
        )

        # Traitement du fichier
        if uploaded_file is not None:
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                if self.example_manager.load_examples(content):
                    st.success(f"Successfully loaded {self.example_manager.status['count']} examples")

                    # Affichage des métriques
                    cols = st.columns(3)
                    status = self.example_manager.get_status()

                    with cols[0]:
                        st.metric("Examples Loaded", status['count'])
                    with cols[1]:
                        st.metric("Status", "✓ Active")
                    with cols[2]:
                        if status['last_update']:
                            st.metric("Last Update", status['last_update'].split('T')[0])

                    # Affichage des exemples
                    st.markdown("### Examples Catalog")
                    catalog = self.example_manager.get_example_catalog()

                    # Sélection d'un exemple spécifique
                    example_titles = [f"Example {i + 1}: {ex['input'][:50]}..."
                                      for i, ex in enumerate(catalog)]
                    selected = st.selectbox(
                        "View Example",
                        example_titles,
                        key="example_selector"
                    )

                    if selected:
                        idx = example_titles.index(selected)
                        example = catalog[idx]

                        st.markdown("#### Selected Example Details")
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown("**Question:**")
                            st.info(example['input'])

                        with col2:
                            st.markdown("**SQL Query:**")
                            st.code(example['query'], language='sql')

                    # Résumé de tous les exemples
                    st.markdown("### All Examples Overview")

                    # Utilisation de tabs au lieu d'expanders imbriqués
                    example_tabs = st.tabs([f"Example {i + 1}" for i in range(len(catalog))])

                    for i, (tab, example) in enumerate(zip(example_tabs, catalog)):
                        with tab:
                            st.markdown("**Question:**")
                            st.info(example['input'])

                            st.markdown("**Query:**")
                            st.code(example['query'], language='sql')

                            # Option copier la requête
                            if st.button(f"Copy Query", key=f"copy_{i}"):
                                st.session_state[f'clipboard_{i}'] = example['query']
                                st.success("Query copied to clipboard!")

                    # Bouton d'analyse
                    st.markdown("### Analysis")
                    col_analyze = st.columns([2, 1])[0]  # Utilise une colonne pour centrer le bouton
                    with col_analyze:
                        if st.button("🔍 Analyze SQL Business Rules",
                                     help="Analyze queries to extract business rules and relationships",
                                     use_container_width=True):
                            self._analyze_queries()

                else:
                    st.error(f"Error loading examples: {self.example_manager.status['error']}")

            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                st.error(f"Error processing file: {str(e)}")

    def _analyze_queries(self):
        """Lance l'analyse LLM des requêtes."""
        with st.spinner("Analyzing business rules in queries..."):
            try:
                analyzer = SQLRulesAnalyzer()
                catalog = self.example_manager.get_example_catalog()

                # Container pour la progression
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    analyses = []
                    for i, example in enumerate(catalog):
                        status_text.write(f"Analyzing query {i + 1}/{len(catalog)}...")
                        analysis = analyzer.analyze_query(example['query'], example['input'])
                        analyses.append(analysis)
                        progress_bar.progress((i + 1) / len(catalog))

                    # Fusion et stockage des résultats
                    merged_analysis = analyzer.merge_analyses(analyses)
                    st.session_state['sql_analysis_results'] = merged_analysis

                    status_text.success("Analysis complete! Switch to the SQL Examples Analysis tab to view results.")

            except Exception as e:
                logger.error(f"Error analyzing queries: {str(e)}")
                st.error(f"Error during analysis: {str(e)}")