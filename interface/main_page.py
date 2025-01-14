# interface/main_page.py

import streamlit as st
import logging
import time
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from database.connection import DatabaseConnection
from database.schema_analyzer import SchemaAnalyzer
from analysis.llm_analyzer import LLMAnalyzer
from visualization.knowledge_graph_view import KnowledgeGraphVisualizer
from interface.results_view import ResultsView
from storage.result_manager import ResultManager
from core.config import Config
from examples.examples_view import ExamplesView
from interface.sql_rules_view import SQLRulesView
from analysis.analysis_merger import AnalysisMerger  # Ajout de l'import
from interface.catalog_view import CatalogView  # Import nécessaire pour le nouveau composant




logger = logging.getLogger(__name__)


class MainPage:
    def __init__(self):
        """Initialise la page principale de l'application."""
        self.init_session_state()
        self.db_connection = DatabaseConnection()
        self.result_manager = ResultManager()
        self.graph_visualizer = KnowledgeGraphVisualizer()
        self.results_view = ResultsView()
        self.apply_custom_styles()
        self.examples_view = ExamplesView()
        self.sql_rules_view = SQLRulesView()
        self.catalog_view = CatalogView()  # Initialisation du nouveau composant

    def init_session_state(self):
        """Initialise l'état de la session Streamlit."""
        session_vars = {
            'analysis_status': 'idle',
            'analysis_progress': {},
            'schema_analysis_results': {},
            'llm_analysis_results': {},
            'analysis_processing_time': 0,
            'engine': None,
            'selected_schemas': [],
            'selected_objects': {},
            'current_connection': None
        }

        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

    def apply_custom_styles(self):
        """Applique les styles CSS personnalisés."""
        custom_css = """
        <style>
            .stButton > button {
                font-size: 16px;
                padding: 4px 20px;
                width: auto;
                height: auto;
            }

            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }

            .progress-container {
                margin: 20px 0;
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def render(self):
        """Affiche la page principale de l'application."""
        st.title("Advanced Database Schema Analyzer")

        # Layout en colonnes
        col1, col2 = st.columns([1, 3])

        with col1:
            self.render_sidebar()

        with col2:
            self.render_main_content()

        # Footer
        st.markdown("---")
        st.markdown("Advanced Database Schema Analyzer - Created with ❤️ by DUKE AI")

    def render_sidebar(self):
        """Affiche la barre latérale avec les contrôles."""
        with st.sidebar:
            with st.expander("Database Connection", expanded=True):
                self.render_connection_section()

            if st.session_state.get('engine'):
                with st.expander("Analysis Configuration", expanded=True):
                    self.render_analysis_config()

            if st.session_state.get('engine'):
                with st.expander("SQL Examples", expanded=True):
                    self.examples_view.render()




    def render_connection_section(self):
        """Affiche la section de connexion à la base de données."""
        try:
            connections = self.db_connection.load_connections()
            selected_conn = st.selectbox(
                "Select Connection",
                ["New Connection"] + list(connections.keys())
            )

            if selected_conn == "New Connection":
                self.render_new_connection_form()
            else:
                self.render_existing_connection_form(connections[selected_conn])

        except Exception as e:
            logger.error(f"Error in connection section: {str(e)}")
            st.error("Error loading connections")

    def render_new_connection_form(self):
        """Affiche le formulaire de nouvelle connexion."""
        conn_name = st.text_input("Connection Name")
        db_type = st.selectbox("Database Type", ["postgresql", "mysql", "mssql"])
        user = st.text_input("User")
        password = st.text_input("Password", type="password")
        host = st.text_input("Host")
        port = st.text_input("Port")
        database = st.text_input("Database")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection"):
                self.test_database_connection(db_type, user, password, host, port, database)

        with col2:
            if st.button("Save Connection"):
                if not conn_name:
                    st.error("Please provide a connection name")
                else:
                    self.save_database_connection(conn_name, {
                        'db_type': db_type,
                        'user': user,
                        'password': password,
                        'host': host,
                        'port': port,
                        'database': database
                    })

    def render_existing_connection_form(self, conn_details: Dict):
        """Affiche le formulaire pour une connexion existante."""
        try:
            # Utilise le nom de la connexion s'il existe, sinon utilise la clé comme nom
            conn_name = conn_details.get('name', '')
            if not conn_name:
                connections = self.db_connection.load_connections()
                for name, details in connections.items():
                    if details == conn_details:
                        conn_name = name
                        break

            conn_name = st.text_input("Connection Name", value=conn_name)
            db_type = st.selectbox(
                "Database Type",
                ["mysql", "postgresql", "mssql"],
                index=["mysql", "postgresql", "mssql"].index(conn_details.get('db_type', 'mysql'))
            )

            user = st.text_input("User", value=conn_details.get('user', ''))
            password = st.text_input("Password", value=conn_details.get('password', ''), type="password")
            host = st.text_input("Host", value=conn_details.get('host', ''))
            port = st.text_input("Port", value=conn_details.get('port', ''))
            database = st.text_input("Database", value=conn_details.get('database', ''))

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Test", key="test_existing_conn",
                             help="Test the database connection"):
                    self.test_database_connection(db_type, user, password, host, port, database)

            with col2:
                if st.button("Update", key="update_existing_conn",
                             help="Update the connection details"):
                    self.save_database_connection(conn_name, {
                        'db_type': db_type,
                        'user': user,
                        'password': password,
                        'host': host,
                        'port': port,
                        'database': database
                    })
                    st.success(f"Connection '{conn_name}' updated successfully")
                    st.rerun()

            with col3:
                if st.button("Delete", key="delete_existing_conn",
                             help="Delete this connection"):
                    self.db_connection.delete_connection(conn_name)
                    st.success(f"Connection '{conn_name}' deleted successfully")
                    st.rerun()

        except Exception as e:
            logger.error(f"Error in render_existing_connection_form: {str(e)}")
            st.error(f"Error loading connection details: {str(e)}")

    def render_analysis_config(self):
        """Affiche la configuration de l'analyse."""
        try:
            if 'engine' not in st.session_state or not st.session_state.engine:
                st.warning("Please establish a database connection first.")
                return

            schema_analyzer = SchemaAnalyzer(st.session_state.engine)
            schemas = schema_analyzer.get_schemas()

            st.session_state.selected_schemas = st.multiselect(
                "Select Schemas",
                options=schemas,
                default=st.session_state.get('selected_schemas', [])
            )

            # Ajout du champ exclude_pattern
            exclude_pattern = st.text_input(
                "Exclude Pattern (optional)",
                help="Patterns to exclude from analysis (e.g., 'temp_*')"
            )

            if st.session_state.selected_schemas:
                st.session_state.selected_objects = self.render_object_selection(
                    schema_analyzer,
                    st.session_state.selected_schemas,
                    exclude_pattern
                )

                if st.button("Start Analysis"):
                    self.start_analysis(
                        schema_analyzer,
                        exclude_pattern=exclude_pattern
                    )

        except Exception as e:
            logger.error(f"Error in analysis config: {str(e)}")
            st.error("Error configuring analysis")

    def render_query_section(self):
        """Affiche la section des exemples de requêtes."""
        uploaded_file = st.file_uploader(
            "Upload Query Examples (JSON)",
            type="json",
            help="Upload a JSON file containing query examples"
        )

        if uploaded_file is not None:
            try:
                query_examples = json.load(uploaded_file)
                if isinstance(query_examples, dict) and 'examples' in query_examples:
                    query_examples = query_examples['examples']

                if isinstance(query_examples, list):
                    st.session_state.query_examples = query_examples
                    st.success(f"Loaded {len(query_examples)} query examples")

                    if query_examples:
                        with st.expander("View first example"):
                            st.json(query_examples[0])
                else:
                    st.error("Invalid query examples format. Expected a list or a dictionary with 'examples' key.")

            except Exception as e:
                logger.error(f"Error processing query file: {str(e)}")
                st.error(f"Error processing query file: {str(e)}")

    def render_object_selection(self, analyzer: SchemaAnalyzer, schemas: List[str], exclude_pattern: str) -> Dict:
        """
        Affiche la sélection des objets pour l'analyse.

        Args:
            analyzer: Instance de SchemaAnalyzer
            schemas: Liste des schémas sélectionnés
            exclude_pattern: Pattern d'exclusion

        Returns:
            Dict des objets sélectionnés
        """
        selected_objects = {}

        for schema in schemas:
            st.write(f"### Objects in {schema}")
            try:
                objects = analyzer.get_objects(schema, exclude_pattern)

                for obj_type, obj_list in objects.items():
                    if obj_list:
                        selected = st.multiselect(
                            f"Select {obj_type}",
                            options=obj_list,
                            key=f"{schema}_{obj_type}"
                        )
                        if selected:
                            if schema not in selected_objects:
                                selected_objects[schema] = {}
                            selected_objects[schema][obj_type] = selected

            except Exception as e:
                logger.error(f"Error getting objects: {str(e)}")
                st.error(f"Error loading objects for {schema}")

        return selected_objects

    def render_main_content(self):
        """Affiche le contenu principal avec les résultats."""
        if st.session_state.analysis_status == 'running':
            self.render_analysis_progress()
        elif st.session_state.schema_analysis_results:
            # Position du bouton de fusion en haut de l'interface
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🔄 Merge Analyses",
                           help="Merge SQL and Schema analyses into complete catalog",
                           use_container_width=True):
                    if 'sql_analysis_results' in st.session_state:
                        st.session_state['merged_catalog'] = AnalysisMerger.merge_analyses(
                            st.session_state.schema_analysis_results,
                            st.session_state.sql_analysis_results
                        )
                        st.success("Analyses merged successfully!")
                    else:
                        st.warning("SQL analysis results not available. Run SQL analysis first.")

            # Création des onglets
            tabs = st.tabs([
                "Data Model Analysis",
                "SQL Examples Analysis",
                "Data Catalog"
            ])

            with tabs[0]:
                self.results_view.render_analysis_results(
                    st.session_state.schema_analysis_results
                )

            with tabs[1]:
                if 'sql_analysis_results' in st.session_state:
                    self.sql_rules_view.render_rules_tab(
                        st.session_state['sql_analysis_results']
                    )
                else:
                    st.info("Run SQL analysis from the sidebar to see results here.")

            with tabs[2]:
                if 'merged_catalog' in st.session_state:
                    self.catalog_view.render_catalog(
                        st.session_state['merged_catalog']
                    )
                else:
                    st.info("Click 'Merge Analyses' button to generate the complete catalog.")
        else:
            st.info("No analysis results to display. Please configure and run an analysis.")


    def render_analysis_progress(self):
        """Affiche la progression de l'analyse."""
        st.write("### Analysis Progress")

        # Calcul de la progression
        total_objects = 0
        processed_objects = 0

        for schema, progress in st.session_state.analysis_progress.items():
            total_objects += progress.get('total', 0)
            processed_objects += progress.get('processed', 0)

        # Barre de progression globale
        if total_objects > 0:
            progress = processed_objects / total_objects
            st.progress(progress)
            st.write(f"Overall Progress: {progress:.1%}")

        # Détails par schéma
        with st.expander("Progress Details", expanded=True):
            for schema, progress in st.session_state.analysis_progress.items():
                st.write(f"**{schema}**")
                st.write(f"- Processed: {progress.get('processed', 0)} / {progress.get('total', 0)}")
                if 'current_object' in progress:
                    st.write(f"- Currently processing: {progress['current_object']}")



    def render_analysis_results(self):
        """Affiche les résultats de l'analyse."""
        st.write("### Analysis Results")

        # Sélection du schéma
        schemas = list(st.session_state.schema_analysis_results.keys())
        selected_schema = st.selectbox("Select Schema", schemas)

        if selected_schema:
            schema_results = st.session_state.schema_analysis_results[selected_schema]

            # Création des onglets
            tabs = st.tabs(["Overview", "Tables", "Knowledge Graph", "Documentation"])

            with tabs[0]:
                self.render_overview_tab(schema_results)

            with tabs[1]:
                self.render_tables_tab(schema_results)

            with tabs[2]:
                self.render_knowledge_graph_tab(schema_results)

            with tabs[3]:
                self.render_documentation_tab(schema_results)

    def test_database_connection(self, db_type: str, user: str, password: str,
                                 host: str, port: str, database: str):
        """Teste la connexion à la base de données."""
        try:
            engine = self.db_connection.get_db_engine(
                db_type, user, password, host, port, database
            )
            if self.db_connection.test_connection(engine):
                st.session_state['engine'] = engine
                st.success("Connection successful!")
            else:
                st.error("Connection failed!")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            st.error(f"Connection error: {str(e)}")

    def save_database_connection(self, name: str, details: Dict):
        """Sauvegarde une connexion à la base de données."""
        try:
            self.db_connection.save_connection(name, details)
            st.success(f"Connection '{name}' saved successfully!")

            # Test optionnel de la connexion après la sauvegarde
            engine = self.db_connection.get_db_engine(
                details['db_type'],
                details['user'],
                details['password'],
                details['host'],
                details['port'],
                details['database']
            )

            if self.db_connection.test_connection(engine):
                st.session_state['engine'] = engine
                st.info("Connection tested and working!")

        except Exception as e:
            logger.error(f"Error saving connection: {str(e)}")
            st.error(f"Error saving connection: {str(e)}")

    def render_object_selection(self, analyzer: SchemaAnalyzer,
                                schemas: List[str], exclude_pattern: str) -> Dict:
        """Affiche et gère la sélection des objets pour l'analyse."""
        selected_objects = {}

        for schema in schemas:
            st.write(f"### Objects in {schema}")
            try:
                objects = analyzer.get_objects(schema, exclude_pattern)

                for obj_type, obj_list in objects.items():
                    if obj_list:  # N'affiche que s'il y a des objets
                        selected = st.multiselect(
                            f"Select {obj_type} from {schema}",
                            options=obj_list,
                            key=f"{schema}_{obj_type}"
                        )
                        if selected:
                            if schema not in selected_objects:
                                selected_objects[schema] = {}
                            selected_objects[schema][obj_type] = selected

            except Exception as e:
                logger.error(f"Error getting objects for schema {schema}: {str(e)}")
                st.error(f"Error loading objects for schema {schema}")

        return selected_objects

    def start_analysis(self, schema_analyzer: SchemaAnalyzer, exclude_pattern: str = None):
        """Lance l'analyse complète incluant l'analyse LLM."""
        try:
            st.session_state.analysis_status = 'running'
            start_time = time.time()

            # Étape 1: Analyse des métadonnées
            schema_results = schema_analyzer.start_analysis(
                st.session_state.selected_schemas,
                st.session_state.selected_objects,
                exclude_pattern
            )

            # Étape 2: Analyse LLM
            llm_analyzer = LLMAnalyzer()

            for schema in st.session_state.selected_schemas:
                for obj_type in ['tables', 'views']:
                    if obj_type in schema_results[schema]:
                        for obj_name, obj_data in schema_results[schema][obj_type].items():
                            try:
                                # Utilisation de analyze_database_object
                                llm_results = llm_analyzer.analyze_database_object(
                                    object_data=obj_data,
                                    object_type=obj_type[:-1]  # Retire le 's' de 'tables'/'views'
                                )

                                # Intégration des résultats
                                if llm_results and 'analysis_results' in llm_results:
                                    schema_results[schema][obj_type][obj_name]['llm_analysis'] = \
                                        llm_results['analysis_results']

                                # Log pour debug
                                logger.info(f"LLM analysis completed for {schema}.{obj_name}")

                            except Exception as e:
                                logger.error(f"Error in LLM analysis for {schema}.{obj_name}: {str(e)}")
                                schema_results[schema][obj_type][obj_name]['llm_analysis_error'] = str(e)

            st.session_state.schema_analysis_results = schema_results

            # Sauvegarde des résultats
            self.result_manager.save_analysis_results(
                schema_results,
                prefix="complete_analysis"
            )

            st.session_state.analysis_processing_time = time.time() - start_time
            st.session_state.analysis_status = 'completed'

            # Log pour debug
            logger.info("Analysis completed successfully")

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            st.error(f"Error during analysis: {str(e)}")
            st.session_state.analysis_status = 'error'

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            st.error(f"Error during analysis: {str(e)}")
            st.session_state.analysis_status = 'error'

    def render_overview_tab(self, schema_results: Dict):
        """Affiche l'onglet de vue d'ensemble."""
        st.write("### Schema Overview")

        # Calcul des statistiques globales
        total_tables = len(schema_results.get('tables', {}))
        total_views = len(schema_results.get('views', {}))
        analysis_duration = schema_results.get('metadata', {}).get('analysis_duration', 0)

        # Affichage des métriques
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Tables", total_tables)
        with col2:
            st.metric("Total Views", total_views)
        with col3:
            st.metric("Analysis Duration", f"{analysis_duration:.2f}s")

        # Ajout des statistiques détaillées si disponibles
        if schema_results.get('tables'):
            st.write("### Table Statistics")

            # Calcul des statistiques globales des tables
            total_rows = 0
            total_columns = 0
            total_indexes = 0
            total_foreign_keys = 0

            for table_name, table_data in schema_results['tables'].items():
                total_rows += table_data.get('row_count', 0)
                total_columns += len(table_data.get('columns', []))
                total_indexes += len(table_data.get('metadata', {}).get('indexes', []))
                total_foreign_keys += len(table_data.get('metadata', {}).get('foreign_keys', []))

            # Affichage des statistiques détaillées
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Total Columns", total_columns)
            with col3:
                st.metric("Total Indexes", total_indexes)
            with col4:
                st.metric("Total Foreign Keys", total_foreign_keys)

    def _render_tables_tab(self, schema_results: Dict[str, Any]):
        """Affiche l'onglet Tables avec les règles fusionnées."""
        try:
            st.header("Tables Analysis")

            # Sélection de la table
            table_names = list(schema_results.get('tables', {}).keys())
            selected_table = st.selectbox("Select Table", table_names, key="catalog_table_select")

            if selected_table:
                table_data = schema_results['tables'][selected_table]

                # Table Description
                st.markdown(f"## {selected_table}")
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
                    for i, rule in enumerate(sql_rules['table_rules']):
                        with st.expander(f"📋 Rule {i + 1}: {rule['description'][:50]}..."):
                            st.markdown(f"**Description:** {rule['description']}")
                            st.markdown(f"**Impact:** {rule.get('impact', 'N/A')}")
                            if 'business_justification' in rule:
                                st.markdown(f"**Justification:** {rule['business_justification']}")
                            if rule.get('source_queries'):
                                st.markdown("**Reference Queries:**")
                                for query in rule['source_queries']:
                                    st.code(query, language='sql')

                # Columns
                st.markdown("### Columns")
                # Sélection de la colonne
                column_names = list(table_data.get('columns', {}).keys())
                selected_column = st.selectbox("Select Column", column_names, key="catalog_column_select")

                if selected_column:
                    col_data = table_data['columns'][selected_column]
                    st.markdown(f"#### {selected_column}")

                    # Information basique de la colonne
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Type:** `{col_data.get('type', 'N/A')}`")
                    with col2:
                        st.markdown(f"**Description:** {col_data.get('description', 'N/A')}")

                    # Règles SQL de la colonne
                    if sql_rules and 'column_rules' in sql_rules:
                        column_rules = sql_rules['column_rules'].get(selected_column, [])
                        if column_rules:
                            st.markdown("#### Business Rules")
                            for i, rule in enumerate(column_rules):
                                with st.expander(f"📋 Rule {i + 1}: {rule['description'][:50]}..."):
                                    st.markdown(f"**Description:** {rule['description']}")
                                    if 'context' in rule:
                                        st.markdown(f"**Context:** {rule['context']}")
                                    if rule.get('source_queries'):
                                        st.markdown("**Reference Queries:**")
                                        for query in rule['source_queries']:
                                            st.code(query, language='sql')

                # Relations
                if sql_rules.get('joins'):
                    st.markdown("### Table Relationships")
                    for i, join in enumerate(sql_rules['joins']):
                        with st.expander(f"🔗 Relationship {i + 1}: {join['source_table']} ↔ {join['target_table']}"):
                            st.markdown(f"**Business Meaning:** {join['business_meaning']}")
                            st.markdown(f"**Importance:** {join.get('importance', 'N/A')}")
                            st.markdown(f"**Justification:** {join.get('justification', 'N/A')}")
                            if join.get('source_queries'):
                                st.markdown("**Reference Queries:**")
                                for query in join['source_queries']:
                                    st.code(query, language='sql')

        except Exception as e:
            logger.error(f"Error rendering tables tab: {str(e)}")
            st.error("Error displaying tables")

    def render_knowledge_graph_tab(self, schema_results: Dict):
        """Affiche l'onglet du graphe de connaissances."""
        st.write("### Knowledge Graph")

        if 'knowledge_graph' in schema_results:
            graph_data = schema_results['knowledge_graph']

            # Options de visualisation
            st.write("#### Visualization Options")
            show_labels = st.checkbox("Show Labels", value=True)
            show_types = st.checkbox("Show Object Types", value=True)

            # Création et affichage du graphe
            try:
                fig = self.graph_visualizer.create_visualization(
                    graph_data['nodes'],
                    graph_data['edges'],
                    show_labels=show_labels,
                    show_types=show_types
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export du graphe
                if st.button("Export Graph"):
                    self.graph_visualizer.export_to_html(fig)
                    st.success("Graph exported successfully!")

            except Exception as e:
                logger.error(f"Error rendering knowledge graph: {str(e)}")
                st.error("Error rendering knowledge graph")
        else:
            st.info("No knowledge graph available for this schema")

    def render_documentation_tab(self, schema_results: Dict):
        """Affiche l'onglet de documentation."""
        st.write("### Technical Documentation")

        if 'technical_documentation' in schema_results:
            doc = schema_results['technical_documentation']

            # Architecture
            if 'architecture_overview' in doc:
                st.write("#### Architecture Overview")
                st.markdown(doc['architecture_overview'])

            # Tables
            if 'table_explanations' in doc:
                st.write("#### Database Objects")
                for table, explanation in doc['table_explanations'].items():
                    with st.expander(table):
                        st.markdown(explanation)

            # Guidelines
            if 'usage_guidelines' in doc:
                st.write("#### Usage Guidelines")
                st.markdown(doc['usage_guidelines'])

            # Optimization
            if 'optimization_suggestions' in doc:
                st.write("#### Optimization Suggestions")
                for suggestion in doc['optimization_suggestions']:
                    st.write(f"- {suggestion}")

            # Export documentation
            if st.button("Export Documentation"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"documentation_{timestamp}.md"

                    self.result_manager.save_documentation(
                        schema_results['technical_documentation'],
                        filename
                    )
                    st.success(f"Documentation exported to {filename}")

                except Exception as e:
                    logger.error(f"Error exporting documentation: {str(e)}")
                    st.error("Error exporting documentation")
        else:
            st.info("No documentation available for this schema")