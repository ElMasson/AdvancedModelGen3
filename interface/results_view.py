# results_view.py - Ajout des imports nécessaires
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Optional, Any, Union
import logging
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from analysis.llm_analyzer import LLMAnalyzer
from analysis.relationship_analyzer import RelationshipAnalyzer
from core.config import Config
from core.utils import JSONEncoder
from visualization.data_model_view import DataModelView

import pandas as pd

logger = logging.getLogger(__name__)

# Définition du composant HTML/JavaScript pour le DataModelView
DATA_MODEL_COMPONENT = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"></script>

<div id="data-model-root"></div>

<script>
    // Le code React compilé sera injecté ici
    const dataModelCode = `{react_code}`;
    const modelData = {model_data};

    // Création d'une fonction wrapper pour le composant React
    function DataModelWrapper() {
        return React.createElement(eval(dataModelCode), modelData);
    }

    // Rendu du composant
    ReactDOM.render(
        React.createElement(DataModelWrapper),
        document.getElementById('data-model-root')
    );
</script>
"""

class ResultsView:
    def __init__(self):
        def __init__(self):
            self.llm_analyzer = LLMAnalyzer()
            self.data_model_view = DataModelView()

    def _render_column_metadata(self, column: Dict[str, Any]):
        """
        Affiche les métadonnées d'une colonne.

        Args:
            column: Données de la colonne
        """
        try:
            if 'statistics' in column:
                stats = column['statistics']

                # Statistiques de base
                st.markdown("**Statistiques de base:**")
                basic_stats = {
                    'total_count': 'Total',
                    'unique_count': 'Valeurs uniques',
                    'null_count': 'Valeurs nulles'
                }
                for key, label in basic_stats.items():
                    if key in stats:
                        st.markdown(f"- {label}: {stats[key]}")

                # Statistiques numériques avec distribution
                if 'numerical' in stats:
                    st.markdown("**Statistiques numériques:**")
                    num_stats = stats['numerical']
                    numerical_stats = {
                        'min': 'Minimum',
                        'max': 'Maximum',
                        'mean': 'Moyenne',
                        'median': 'Médiane',
                        'q1': 'Premier quartile',
                        'q3': 'Troisième quartile',
                        'std_dev': 'Écart type',
                        'variance': 'Variance'
                    }
                    for key, label in numerical_stats.items():
                        if key in num_stats and num_stats[key] is not None:
                            st.markdown(f"- {label}: {num_stats[key]:.2f}")

                    # Distribution
                    if 'distribution' in stats:
                        st.markdown("**Distribution:**")
                        buckets = stats['distribution']
                        for bucket in buckets:
                            st.markdown(
                                f"- Bucket {bucket['bucket']}: {bucket['frequency']} valeurs "
                                f"(entre {bucket['min_value']:.2f} et {bucket['max_value']:.2f})"
                            )

                # Statistiques textuelles
                elif 'text_stats' in stats:
                    st.markdown("**Statistiques textuelles:**")
                    text_stats = stats['text_stats']
                    for key, value in text_stats.items():
                        if value is not None:
                            st.markdown(f"- {key.replace('_', ' ').title()}: {value}")

                # Statistiques de dates
                elif 'date_stats' in stats:
                    st.markdown("**Statistiques temporelles:**")
                    date_stats = stats['date_stats']
                    for key, value in date_stats.items():
                        if value is not None:
                            st.markdown(f"- {key.replace('_', ' ').title()}: {value}")

                # Échantillons de valeurs
                if 'sample_values' in stats:
                    st.markdown("**Exemples de valeurs:**")
                    for value in stats['sample_values']:
                        st.markdown(f"- {value}")

        except Exception as e:
            logger.error(f"Error rendering column metadata: {str(e)}")
            st.error("Erreur lors de l'affichage des métadonnées de la colonne")

    def render_analysis_results(self, analysis_results: Dict[str, Any]):
        """Affiche les résultats d'analyse selon le format spécifié."""
        try:
            if not analysis_results:
                st.info("Aucun résultat d'analyse disponible.")
                return

            # Sélection du schéma
            schemas = list(analysis_results.keys())
            selected_schema = st.selectbox("Sélectionner un schéma", schemas)

            if selected_schema:
                schema_results = analysis_results[selected_schema]

                # Création des onglets - Définir explicitement la liste des tabs
                tab_titles = ["Overview", "Tables", "Views", "Data Model"]
                overview_tab, tables_tab, views_tab, datamodel_tab = st.tabs(tab_titles)

                # Onglet Overview
                with overview_tab:
                    self._render_overview_tab(schema_results)

                # Onglet Tables
                with tables_tab:
                    self._render_tables_tab(schema_results)

                # Onglet Views
                with views_tab:
                    self._render_views_tab(schema_results)

                # Onglet Data Model
                with datamodel_tab:
                    self._render_data_model_tab(schema_results)

        except Exception as e:
            logger.error(f"Error rendering analysis results: {str(e)}")
            st.error("Erreur lors de l'affichage des résultats")

    def _render_data_model_tab(self, schema_results: Dict):
        """Affiche l'onglet du modèle de données."""
        try:
            st.header("Modèle de Données")

            if not hasattr(self, 'data_model_view'):
                from visualization.data_model_view import DataModelView
                self.data_model_view = DataModelView()

            # Préparation des données
            model_data = self._prepare_schema_data(schema_results)

            # Affichage du modèle
            self.data_model_view.render(model_data)

        except Exception as e:
            logger.error(f"Error rendering data model tab: {str(e)}")
            st.error("Erreur lors de l'affichage du modèle de données")

    def _prepare_schema_data(self, schema_results: Dict) -> Dict:
        """
        Prépare les données du schéma pour la visualisation.

        Args:
            schema_results: Résultats d'analyse du schéma

        Returns:
            Dict contenant les données formatées pour la visualisation
        """
        try:
            nodes = []
            edges = []

            # Traitement des tables
            if 'tables' in schema_results:
                for table_name, table_data in schema_results['tables'].items():
                    # Ajout du nœud pour la table
                    nodes.append({
                        'id': table_name,
                        'label': table_name,
                        'type': 'table'
                    })

                    # Traitement des relations si elles existent
                    if 'relationships' in table_data:
                        relationships = table_data['relationships']

                        # Relations explicites
                        for rel in relationships.get('explicit_relations', []):
                            edges.append({
                                'source': rel['source'],
                                'target': rel['target'],
                                'type': 'explicit',
                                'label': rel.get('type', 'relation')
                            })

                        # Relations potentielles
                        for rel in relationships.get('potential_relations', []):
                            edges.append({
                                'source': rel['source'],
                                'target': rel['target'],
                                'type': 'potential',
                                'label': rel.get('type', 'potential')
                            })

            return {
                'nodes': nodes,
                'edges': edges
            }

        except Exception as e:
            logger.error(f"Error preparing schema data: {str(e)}")
            return {'nodes': [], 'edges': []}

    def _render_views_tab(self, schema_results: Dict):
        """
        Affiche l'onglet des vues.

        Args:
            schema_results: Résultats d'analyse du schéma
        """
        try:
            st.header("Vues")

            if 'views' in schema_results and schema_results['views']:
                # Sélection de la vue
                selected_view = st.selectbox(
                    "Sélectionner une vue",
                    list(schema_results['views'].keys())
                )

                if selected_view:
                    self._render_detailed_analysis(
                        schema_results['views'][selected_view],
                        selected_view
                    )
            else:
                st.info("Aucune vue disponible pour analyse.")

        except Exception as e:
            logger.error(f"Error rendering views tab: {str(e)}")
            st.error("Erreur lors de l'affichage des vues")

    def _render_tables_tab(self, schema_results: Dict):
        """
        Affiche l'onglet des tables.

        Args:
            schema_results: Résultats d'analyse du schéma
        """
        try:
            st.header("Tables")

            if 'tables' in schema_results and schema_results['tables']:
                # Sélection de la table
                selected_table = st.selectbox(
                    "Sélectionner une table",
                    list(schema_results['tables'].keys())
                )

                if selected_table:
                    self._render_detailed_analysis(
                        schema_results['tables'][selected_table],
                        selected_table
                    )
            else:
                st.info("Aucune table disponible pour analyse.")

        except Exception as e:
            logger.error(f"Error rendering tables tab: {str(e)}")
            st.error("Erreur lors de l'affichage des tables")

    def _render_overview_tab(self, schema_results: Dict):
        """
        Affiche l'onglet Overview avec les statistiques globales.

        Args:
            schema_results: Résultats d'analyse du schéma
        """
        st.header("Aperçu Global")

        # Récupération du temps d'analyse
        analysis_time = self._get_analysis_time(schema_results)

        # Statistiques de base
        col1, col2, col3, col4 = st.columns(4)

        # Nombre de tables
        with col1:
            table_count = len(schema_results.get('tables', {}))
            st.metric("Nombre de Tables", table_count)

        # Nombre total de lignes
        with col2:
            total_rows = self._calculate_total_rows(schema_results)
            st.metric("Total des Lignes", f"{total_rows:,}")

        # Temps de traitement avec unité adaptative
        with col3:
            if analysis_time < 60:
                time_str = f"{analysis_time:.2f}s"
            else:
                minutes = int(analysis_time // 60)
                seconds = analysis_time % 60
                time_str = f"{minutes}m {seconds:.2f}s"
            st.metric("Temps d'Analyse", time_str)

        # Nombre de vues
        with col4:
            view_count = len(schema_results.get('views', {}))
            st.metric("Nombre de Vues", view_count)

        # Graphique de distribution des tailles de tables
        st.subheader("Distribution des Tailles de Tables")
        table_sizes = {
            name: table.get('row_count', 0)
            for name, table in schema_results.get('tables', {}).items()
        }
        if table_sizes:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(table_sizes.keys()),
                    y=list(table_sizes.values()),
                    text=list(table_sizes.values()),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Nombre de Lignes par Table",
                xaxis_title="Tables",
                yaxis_title="Nombre de Lignes",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistiques avancées
        st.subheader("Statistiques Avancées")

        col1, col2 = st.columns(2)

        with col1:
            # Types de données utilisés
            data_types = self._get_data_type_statistics(schema_results)
            if data_types:
                fig = px.pie(
                    values=list(data_types.values()),
                    names=list(data_types.keys()),
                    title="Distribution des Types de Données"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Statistiques de nullité
            null_stats = self._get_null_statistics(schema_results)
            if null_stats:
                fig = px.bar(
                    x=list(null_stats.keys()),
                    y=list(null_stats.values()),
                    title="Pourcentage de Valeurs Nulles par Table"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _calculate_total_rows(self, schema_results: Dict) -> int:
        """
        Calcule le nombre total de lignes dans toutes les tables.

        Args:
            schema_results: Résultats d'analyse du schéma

        Returns:
            int: Nombre total de lignes
        """
        total = 0
        try:
            for table in schema_results.get('tables', {}).values():
                # Vérification des différents emplacements possibles pour le compte de lignes
                row_count = (
                        table.get('row_count') or
                        table.get('metadata', {}).get('row_count') or
                        table.get('statistics', {}).get('total_rows', 0)
                )
                total += row_count
            return total
        except Exception as e:
            logger.error(f"Error calculating total rows: {str(e)}")
            return 0

    def _render_table_metrics(self, table_data: Dict):
        """
        Affiche les métriques principales d'une table.

        Args:
            table_data: Données de la table
        """
        try:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Nombre de Colonnes",
                    len(table_data.get('columns', []))
                )

            with col2:
                row_count = (
                        table_data.get('row_count') or
                        table_data.get('metadata', {}).get('row_count', 0)
                )
                st.metric("Nombre de Lignes", f"{row_count:,}")

            with col3:
                null_ratio = self._calculate_null_ratio(table_data)
                st.metric(
                    "Ratio de Nulls",
                    f"{null_ratio:.2%}"
                )

            with col4:
                unique_ratio = self._calculate_unique_ratio(table_data)
                st.metric(
                    "Ratio d'Unicité",
                    f"{unique_ratio:.2%}"
                )

        except Exception as e:
            logger.error(f"Error rendering table metrics: {str(e)}")
            st.error("Erreur lors de l'affichage des métriques de la table")

    def _calculate_unique_ratio(self, table_data: Dict) -> float:
        """
        Calcule le ratio de valeurs uniques dans la table.

        Args:
            table_data: Données de la table

        Returns:
            float: Ratio de valeurs uniques
        """
        try:
            total_unique = 0
            total_values = 0
            for column in table_data.get('columns', []):
                if 'statistics' in column:
                    stats = column['statistics']
                    total_unique += stats.get('unique_count', 0)
                    total_values += stats.get('total_count', 0)

            return total_unique / total_values if total_values > 0 else 0

        except Exception as e:
            logger.error(f"Error calculating unique ratio: {str(e)}")
            return 0.0

    def _calculate_null_ratio(self, table_data: Dict) -> float:
        """
        Calcule le ratio de valeurs nulles dans la table.

        Args:
            table_data: Données de la table

        Returns:
            float: Ratio de valeurs nulles
        """
        try:
            total_nulls = 0
            total_values = 0
            for column in table_data.get('columns', []):
                if 'statistics' in column:
                    stats = column['statistics']
                    total_nulls += stats.get('null_count', 0)
                    total_values += stats.get('total_count', 0)

            return total_nulls / total_values if total_values > 0 else 0

        except Exception as e:
            logger.error(f"Error calculating null ratio: {str(e)}")
            return 0.0

    def _get_analysis_time(self, schema_results: Dict) -> float:
        """
        Récupère le temps d'analyse depuis les différentes sources possibles.

        Args:
            schema_results: Résultats d'analyse du schéma

        Returns:
            float: Temps d'analyse en secondes
        """
        try:
            # 1. Vérification de la session Streamlit
            if 'analysis_processing_time' in st.session_state:
                return st.session_state.analysis_processing_time

            # 2. Vérification des métadonnées du schéma
            if 'metadata' in schema_results:
                meta = schema_results['metadata']
                # Vérification directe
                if 'analysis_duration' in meta:
                    return float(meta['analysis_duration'])
                # Vérification des timestamps
                if 'timestamp' in meta:
                    end_time = datetime.fromisoformat(meta['timestamp'])
                    if 'start_timestamp' in meta:
                        start_time = datetime.fromisoformat(meta['start_timestamp'])
                        return (end_time - start_time).total_seconds()

            # 3. Vérification dans les résultats des tables
            total_time = 0
            for table_name, table_data in schema_results.get('tables', {}).items():
                if 'llm_analysis' in table_data:
                    llm_meta = table_data['llm_analysis'].get('metadata', {})
                    if 'analysis_duration' in llm_meta:
                        total_time += float(llm_meta['analysis_duration'])
                    elif 'timestamp' in llm_meta and 'start_timestamp' in llm_meta:
                        start = datetime.fromisoformat(llm_meta['start_timestamp'])
                        end = datetime.fromisoformat(llm_meta['timestamp'])
                        total_time += (end - start).total_seconds()

            return total_time if total_time > 0 else st.session_state.get('analysis_processing_time', 0.0)

        except Exception as e:
            logger.error(f"Error getting analysis time: {str(e)}")
            return st.session_state.get('analysis_processing_time', 0.0)

    def _render_detailed_analysis(self, object_data: Dict[str, Any], object_name: str):
        """Affiche l'analyse détaillée d'un objet."""
        try:
            # 1. Nom de l'objet
            st.markdown(f"## {object_name}")

            # 2. Analyse LLM au niveau table
            if 'llm_analysis' in object_data and 'structure_analysis' in object_data['llm_analysis']:
                st.markdown("### Analyse Globale")
                structure_analysis = object_data['llm_analysis']['structure_analysis']
                self._render_llm_table_analysis(structure_analysis)

                # 3. Phases d'analyse LLM
                st.markdown("### Analyses Avancées")

                # Phase 1: Descriptive
                with st.expander("Phase 1: Analyse Descriptive", expanded=True):
                    descriptive = self._get_descriptive_analysis(object_data)
                    for key, value in descriptive.items():
                        st.markdown(f"**{key}:**")
                        if isinstance(value, list):
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(value)

                # Phase 2: Patterns
                with st.expander("Phase 2: Analyse des Patterns", expanded=True):
                    patterns = self._get_pattern_analysis(object_data)
                    for key, value in patterns.items():
                        st.markdown(f"**{key}:**")
                        if isinstance(value, list):
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(str(value))

                # Phase 3: Documentation
                with st.expander("Phase 3: Documentation", expanded=True):
                    doc = self._get_documentation_analysis(object_data)
                    for key, value in doc.items():
                        st.markdown(f"**{key}:**")
                        if isinstance(value, list):
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(str(value))

                # Phase 4: Optimisation
                with st.expander("Phase 4: Optimisation", expanded=True):
                    opt = self._get_optimization_analysis(object_data)
                    for key, value in opt.items():
                        st.markdown(f"**{key}:**")
                        if isinstance(value, list):
                            for item in value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(str(value))

            # 4. Analyse des colonnes
            st.markdown("### Colonnes")
            if 'columns' in object_data:
                for column in object_data['columns']:
                    with st.expander(f"📊 {column['name']} ({column['type']})"):
                        self._render_column_analysis(column, object_data.get('llm_analysis', {}))

        except Exception as e:
            logger.error(f"Error in detailed analysis: {str(e)}")
            st.error("Erreur lors de l'affichage de l'analyse détaillée")

    def _render_llm_table_analysis(self, analysis: Dict):
        """Affiche l'analyse LLM de la table."""
        # Catégories
        if 'categories' in analysis:
            st.markdown("**Catégories:**")
            for category in analysis['categories']:
                st.markdown(f"- {category}")

        # Noms alternatifs
        if 'alternative_names' in analysis:
            st.markdown("**Noms alternatifs:**")
            for name in analysis['alternative_names']:
                st.markdown(f"- {name}")

        # Objectif fonctionnel
        if 'purpose' in analysis:
            st.markdown("**Objectif fonctionnel:**")
            st.markdown(analysis['purpose'])

        # Description fonctionnelle
        if 'functional_description' in analysis:
            st.markdown("**Description fonctionnelle:**")
            st.markdown(analysis['functional_description'])

        # Cas d'utilisation
        if 'use_cases' in analysis:
            st.markdown("**Cas d'utilisation:**")
            for case in analysis['use_cases']:
                st.markdown(f"- {case}")

        # Suggestions
        if 'suggestions' in analysis:
            st.markdown("**Suggestions:**")
            for suggestion in analysis['suggestions']:
                st.markdown(f"- {suggestion}")

    def _render_vertical_histogram(self, column: Dict):
        """Crée un histogramme vertical pour les colonnes numériques."""
        try:
            if 'statistics' in column and 'distribution' in column['statistics']:
                data = column['statistics']['distribution']

                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"[{d['min_value']:.2f}, {d['max_value']:.2f}]" for d in data],
                        y=[d['frequency'] for d in data],
                        text=[d['frequency'] for d in data],
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title=f"Distribution des valeurs - {column['name']}",
                    xaxis_title="Intervalles de valeurs",
                    yaxis_title="Fréquence",
                    bargap=0.1,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            st.error("Erreur lors de la création de l'histogramme")

    def _is_numeric_type(self, column_type: str) -> bool:
        """Vérifie si le type de colonne est numérique."""
        numeric_types = {'int', 'float', 'double', 'decimal', 'numeric', 'real'}
        return any(t in column_type.lower() for t in numeric_types)

    def _get_data_type_statistics(self, schema_results: Dict) -> Dict:
        """
        Calcule les statistiques des types de données.

        Args:
            schema_results: Résultats d'analyse du schéma

        Returns:
            Dict contenant les statistiques des types de données
        """
        data_types = {}
        for table in schema_results.get('tables', {}).values():
            for column in table.get('columns', []):
                data_type = column.get('type', 'unknown')
                data_types[data_type] = data_types.get(data_type, 0) + 1
        return data_types

    def _get_null_statistics(self, schema_results: Dict) -> Dict:
        """
        Calcule les statistiques de nullité.

        Args:
            schema_results: Résultats d'analyse du schéma

        Returns:
            Dict contenant les statistiques de nullité
        """
        null_stats = {}
        for table_name, table in schema_results.get('tables', {}).items():
            total_nulls = 0
            total_values = 0
            for column in table.get('columns', []):
                if 'statistics' in column:
                    stats = column['statistics']
                    total_nulls += stats.get('null_count', 0)
                    total_values += stats.get('total_count', 0)
            if total_values > 0:
                null_stats[table_name] = (total_nulls / total_values) * 100
        return null_stats

    def _render_column_analysis(self, column: Dict, llm_analysis: Dict):
        """
        Affiche l'analyse d'une colonne avec visualisations améliorées.

        Args:
            column: Données de la colonne
            llm_analysis: Analyse LLM
        """
        try:
            # Description LLM
            if 'column_analyses' in llm_analysis:
                col_analysis = llm_analysis['column_analyses'].get(column['name'], {})
                if 'business_purpose' in col_analysis:
                    st.markdown("#### Description")
                    st.markdown(col_analysis['business_purpose'])

            # Métadonnées et statistiques
            st.markdown("#### Métadonnées")
            self._render_column_metadata(column)

            # Distribution des valeurs
            if self._is_numeric_type(column['type']):
                # Pour les colonnes numériques
                st.markdown("#### Distribution")
                self._render_vertical_histogram(column)
            else:
                # Pour les colonnes non numériques
                st.markdown("#### Distribution des Valeurs")
                self._render_categorical_distribution(column)

            # Reste de l'analyse LLM
            if col_analysis:
                self._render_column_llm_analysis(col_analysis)

        except Exception as e:
            logger.error(f"Error rendering column analysis: {str(e)}")
            st.error(f"Erreur lors de l'affichage de l'analyse de la colonne {column['name']}")

    def _render_categorical_distribution(self, column: Dict):
        """
        Affiche la distribution des valeurs pour les colonnes non numériques.

        Args:
            column: Données de la colonne
        """
        try:
            if 'statistics' in column:
                stats = column['statistics']

                # Préparation des données pour le top 10
                top_values = []

                # Vérification des différentes structures possibles
                if 'categories' in stats:
                    # Format direct
                    top_values = stats['categories'][:10]
                elif 'value_counts' in stats:
                    # Format alternatif
                    top_values = [
                                     {
                                         'value': value,
                                         'count': count,
                                         'percentage': (count / stats.get('total_count', 1)) * 100
                                     }
                                     for value, count in stats['value_counts'].items()
                                 ][:10]
                elif isinstance(stats.get('sample_values'), list):
                    # Format avec échantillons
                    samples = stats['sample_values']
                    if isinstance(samples[0], dict):
                        top_values = samples[:10]
                    else:
                        # Conversion des échantillons simples en format attendu
                        from collections import Counter
                        counts = Counter(samples)
                        total = len(samples)
                        top_values = [
                            {
                                'value': value,
                                'count': count,
                                'percentage': (count / total) * 100
                            }
                            for value, count in counts.most_common(10)
                        ]

                if top_values:
                    st.markdown("**Top 10 des valeurs les plus fréquentes:**")

                    # Création du DataFrame pour l'affichage
                    df_data = [
                        {
                            'Valeur': str(item.get('value', '')),
                            'Fréquence': item.get('count', 0),
                            'Pourcentage': f"{item.get('percentage', 0):.2f}%"
                        }
                        for item in top_values
                    ]

                    df = pd.DataFrame(df_data)
                    st.dataframe(df)

                    # Création de l'histogramme vertical
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[str(d['Valeur']) for d in df_data],
                            y=[d['Fréquence'] for d in df_data],
                            text=[d['Pourcentage'] for d in df_data],
                            textposition='auto',
                        )
                    ])

                    fig.update_layout(
                        title=f"Distribution des Top 10 Valeurs - {column['name']}",
                        xaxis_title="Valeurs",
                        yaxis_title="Fréquence",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Affichage des exemples de valeurs
                if 'sample_values' in stats:
                    st.markdown("**Exemples de valeurs:**")
                    samples = stats['sample_values']
                    if isinstance(samples[0], dict):
                        examples = [str(s.get('value', '')) for s in samples[:5]]
                    else:
                        examples = [str(s) for s in samples[:5]]
                    st.markdown(", ".join(examples))

                # Statistiques textuelles si disponibles
                if 'text_stats' in stats:
                    text_stats = stats['text_stats']
                    cols = st.columns(4)

                    with cols[0]:
                        st.metric("Longueur moyenne", f"{text_stats.get('avg_length', 0):.2f}")
                    with cols[1]:
                        st.metric("Longueur min", text_stats.get('min_length', 0))
                    with cols[2]:
                        st.metric("Longueur max", text_stats.get('max_length', 0))
                    with cols[3]:
                        st.metric("Valeurs uniques", text_stats.get('distinct_count', 0))

        except Exception as e:
            logger.error(f"Error in categorical distribution: {str(e)}")
            st.error("Erreur lors de l'affichage de la distribution catégorielle")


    def _get_descriptive_analysis(self, object_data: Dict) -> Dict:
        """
        Extrait l'analyse descriptive des données.

        Args:
            object_data: Données de l'objet

        Returns:
            Dict contenant l'analyse descriptive
        """
        try:
            if 'llm_analysis' in object_data:
                analysis = object_data['llm_analysis']['structure_analysis']
                return {
                    'Objectif': analysis.get('purpose', ''),
                    'Description fonctionnelle': analysis.get('functional_description', ''),
                    'Catégories': analysis.get('categories', []),
                    'Noms alternatifs': analysis.get('alternative_names', []),
                    'Détails techniques': {
                        'Description': analysis.get('detailed_analysis', {}).get('summary', ''),
                        'Points clés': analysis.get('detailed_analysis', {}).get('key_points', []),
                        'Relations': analysis.get('detailed_analysis', {}).get('data_structure', {}).get(
                            'relationships', [])
                    }
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting descriptive analysis: {str(e)}")
            return {}

    def _get_pattern_analysis(self, object_data: Dict) -> Dict:
        """
        Extrait l'analyse des patterns des données.

        Args:
            object_data: Données de l'objet

        Returns:
            Dict contenant l'analyse des patterns
        """
        try:
            if 'llm_analysis' in object_data:
                detailed = object_data['llm_analysis']['structure_analysis'].get('detailed_analysis', {})
                data_structure = detailed.get('data_structure', {})
                return {
                    'Patterns identifiés': detailed.get('usage_patterns', []),
                    'Structure des données': {
                        'Description': data_structure.get('description', ''),
                        'Relations': data_structure.get('relationships', []),
                        'Contraintes': data_structure.get('constraints', [])
                    },
                    'Points clés': detailed.get('key_points', []),
                    'Patterns d\'utilisation': detailed.get('usage_patterns', []),
                    'Analyse d\'intégrité': data_structure.get('integrity', '')
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting pattern analysis: {str(e)}")
            return {}

    def _get_documentation_analysis(self, object_data: Dict) -> Dict:
        """
        Extrait la documentation des données.

        Args:
            object_data: Données de l'objet

        Returns:
            Dict contenant la documentation
        """
        try:
            if 'llm_analysis' in object_data:
                analysis = object_data['llm_analysis']['structure_analysis']
                detailed = analysis.get('detailed_analysis', {})
                return {
                    'Description technique': detailed.get('summary', ''),
                    'Cas d\'utilisation': analysis.get('use_cases', []),
                    'Contraintes': detailed.get('data_structure', {}).get('constraints', []),
                    'Aspects techniques': {
                        'Structure': detailed.get('data_structure', {}).get('description', ''),
                        'Intégrité': detailed.get('data_structure', {}).get('integrity', ''),
                        'Points clés': detailed.get('key_points', [])
                    },
                    'Recommandations': analysis.get('suggestions', [])
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting documentation analysis: {str(e)}")
            return {}

    def _get_optimization_analysis(self, object_data: Dict) -> Dict:
        """
        Extrait l'analyse d'optimisation des données.

        Args:
            object_data: Données de l'objet

        Returns:
            Dict contenant l'analyse d'optimisation
        """
        try:
            if 'llm_analysis' in object_data:
                analysis = object_data['llm_analysis']['structure_analysis']
                detailed = analysis.get('detailed_analysis', {})
                data_structure = detailed.get('data_structure', {})

                return {
                    'Suggestions d\'optimisation': analysis.get('suggestions', []),
                    'Améliorations possibles': detailed.get('key_points', []),
                    'Impact performance': {
                        'Intégrité': data_structure.get('integrity', ''),
                        'Contraintes': data_structure.get('constraints', []),
                        'Patterns d\'utilisation': detailed.get('usage_patterns', [])
                    },
                    'Recommandations techniques': {
                        'Structure': data_structure.get('description', ''),
                        'Relations': data_structure.get('relationships', []),
                        'Points d\'attention': detailed.get('key_points', [])
                    },
                    'Plan d\'action': [
                        "1. Analyse des performances actuelles",
                        "2. Identification des goulots d'étranglement",
                        "3. Optimisation des index et requêtes",
                        "4. Tests de performance",
                        "5. Validation et déploiement"
                    ]
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting optimization analysis: {str(e)}")
            return {}

    def _render_column_llm_analysis(self, column_analysis: Dict):
        """
        Affiche l'analyse LLM d'une colonne.

        Args:
            column_analysis: Analyse LLM de la colonne
        """
        try:
            if 'data_characteristics' in column_analysis:
                st.markdown("#### Caractéristiques des données")
                chars = column_analysis['data_characteristics']
                st.markdown(f"**Nature:** {chars.get('nature', 'N/A')}")
                st.markdown(f"**Valeurs attendues:** {chars.get('expected_values', 'N/A')}")

                if 'special_cases' in chars and chars['special_cases']:
                    st.markdown("**Cas spéciaux:**")
                    for case in chars['special_cases']:
                        st.markdown(f"- {case}")

            if 'business_rules' in column_analysis:
                st.markdown("#### Règles métier")
                for rule in column_analysis['business_rules']:
                    st.markdown(f"- {rule}")

            if 'data_quality' in column_analysis:
                st.markdown("#### Qualité des données")
                quality = column_analysis['data_quality']

                if 'critical_aspects' in quality:
                    st.markdown("**Aspects critiques:**")
                    for aspect in quality['critical_aspects']:
                        st.markdown(f"- {aspect}")

                if 'validation_rules' in quality:
                    st.markdown("**Règles de validation:**")
                    for rule in quality['validation_rules']:
                        st.markdown(f"- {rule}")

                if 'recommendations' in quality:
                    st.markdown("**Recommandations:**")
                    for rec in quality['recommendations']:
                        st.markdown(f"- {rec}")

            if 'relationships' in column_analysis:
                st.markdown("#### Relations")
                rels = column_analysis['relationships']

                if 'dependencies' in rels:
                    st.markdown("**Dépendances:**")
                    for dep in rels['dependencies']:
                        st.markdown(f"- {dep}")

                if 'impact' in rels:
                    st.markdown("**Impact:**")
                    for imp in rels['impact']:
                        st.markdown(f"- {imp}")

        except Exception as e:
            logger.error(f"Error rendering column LLM analysis: {str(e)}")
            st.error("Erreur lors de l'affichage de l'analyse LLM de la colonne")