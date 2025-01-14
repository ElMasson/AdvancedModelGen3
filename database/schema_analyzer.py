# database/schema_analyzer.py
from sqlalchemy import text, inspect
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time
import numpy as np
from scipy import stats
from decimal import Decimal
from analysis.llm_analyzer import LLMAnalyzer
from analysis.relationship_analyzer import RelationshipAnalyzer

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """Analyseur complet de schéma de base de données."""

    def __init__(self, engine):
        """
        Initialise l'analyseur.

        Args:
            engine: SQLAlchemy engine connecté à la base de données
        """
        self.engine = engine
        self.inspector = inspect(engine)
        self.llm_analyzer = LLMAnalyzer()  # Réajout de cette ligne importante
        self._setup_dialect_specifics()

    def _setup_dialect_specifics(self):
        """Configure les requêtes spécifiques au dialecte."""
        dialect = self.engine.dialect.name
        if dialect == 'postgresql':
            self.queries = {
                'column_stats': """
                    SELECT 
                        a.attname as column_name,
                        pg_stats.null_frac * 100 as null_percentage,
                        pg_stats.n_distinct,
                        pg_stats.most_common_vals,
                        pg_stats.most_common_freqs,
                        pg_stats.histogram_bounds
                    FROM pg_stats 
                    JOIN pg_attribute a ON a.attname = pg_stats.attname
                    WHERE schemaname = :schema 
                    AND tablename = :table
                """
            }
        else:
            self.queries = {'column_stats': None}

    def get_schemas(self) -> List[str]:
        """
        Récupère la liste des schémas disponibles.

        Returns:
            Liste des noms de schémas
        """
        try:
            return self.inspector.get_schema_names()
        except Exception as e:
            logger.error(f"Error getting schemas: {str(e)}")
            raise

    def _get_analysis_sample(self, schema: str, table_name: str, sample_size: int = 10000) -> List[Dict]:
        """
        Extrait un échantillon aléatoire pour l'analyse LLM.
        """
        try:
            with self.engine.connect() as connection:
                query = text(f"""
                    SELECT *
                    FROM {schema}.{table_name}
                    ORDER BY RANDOM()
                    LIMIT :sample_size
                """)

                result = connection.execute(query, {"sample_size": sample_size})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result]

        except Exception as e:
            logger.error(f"Error getting analysis sample: {str(e)}")
            return []

    def get_objects(self, schema: str, exclude_pattern: str = "") -> Dict[str, List[str]]:
        """
        Récupère les objets d'un schéma.

        Args:
            schema: Nom du schéma
            exclude_pattern: Pattern d'exclusion

        Returns:
            Dict contenant les listes des tables et vues
        """
        try:
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = :schema 
                AND table_type = 'BASE TABLE'
            """)

            views_query = text("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = :schema
            """)

            params = {"schema": schema}
            if exclude_pattern:
                exclude_pattern = f"%{exclude_pattern}%"
                tables_query = text(f"{tables_query.text} AND table_name NOT LIKE :exclude_pattern")
                views_query = text(f"{views_query.text} AND table_name NOT LIKE :exclude_pattern")
                params["exclude_pattern"] = exclude_pattern

            with self.engine.connect() as connection:
                tables = connection.execute(tables_query, params).fetchall()
                views = connection.execute(views_query, params).fetchall()

            return {
                "tables": [table[0] for table in tables],
                "views": [view[0] for view in views]
            }

        except Exception as e:
            logger.error(f"Error getting objects for schema {schema}: {str(e)}")
            raise

    def analyze_table(self, schema: str, table_name: str) -> Dict[str, Any]:
        try:
            # Code existant pour l'analyse de base
            table_analysis = {
                'metadata': {
                    'primary_keys': self.inspector.get_pk_constraint(table_name, schema),
                    'foreign_keys': self._get_foreign_keys(schema, table_name),
                    'indexes': self._get_indexes(schema, table_name)
                },
                'columns': []
            }

            # Analyse des colonnes existante
            for col in self.inspector.get_columns(table_name, schema):
                column_info = {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default', None))
                }

                stats = self._get_column_statistics(schema, table_name, col['name'], str(col['type']))
                if stats:
                    column_info['statistics'] = stats

                table_analysis['columns'].append(column_info)

            # Ajout des relations via RelationshipAnalyzer
            relationship_analyzer = RelationshipAnalyzer(self.llm_analyzer)  # Utilise le même LLM analyzer
            relationships = relationship_analyzer.analyze_schema_relationships({
                'tables': {table_name: table_analysis},
                'schema': schema
            })
            table_analysis['relationships'] = relationships

            # Analyse LLM existante
            try:
                llm_analysis = self.llm_analyzer.analyze_table(table_analysis)
                if llm_analysis:
                    table_analysis['llm_analysis'] = llm_analysis
            except Exception as e:
                logger.error(f"Error during LLM analysis: {str(e)}")
                table_analysis['llm_analysis_error'] = str(e)

            # Statistiques de base existantes
            with self.engine.connect() as connection:
                count_query = text(f"SELECT COUNT(*) FROM {schema}.{table_name}")
                table_analysis['row_count'] = connection.execute(count_query).scalar()

            table_analysis['analysis_timestamp'] = datetime.now().isoformat()

            return table_analysis

        except Exception as e:
            logger.error(f"Error analyzing table {schema}.{table_name}: {str(e)}")
            raise

    def analyze_view(self, schema: str, view_name: str) -> Dict[str, Any]:
        """
        Analyse optimisée d'une vue.

        Args:
            schema: Nom du schéma
            view_name: Nom de la vue

        Returns:
            Dict contenant les métadonnées essentielles de la vue
        """
        try:
            view_analysis = {
                'metadata': {},
                'columns': [],
                'statistics': {},
                'analysis_timestamp': datetime.now().isoformat()
            }

            with self.engine.connect() as connection:
                # 1. Récupération du code SQL de la vue (requête simplifiée)
                view_def_query = text("""
                    SELECT view_definition
                    FROM information_schema.views
                    WHERE table_schema = :schema
                    AND table_name = :view_name
                """)

                view_def = connection.execute(
                    view_def_query,
                    {"schema": schema, "view_name": view_name}
                ).scalar()

                if view_def:
                    view_analysis['metadata']['definition'] = view_def

                # 2. Analyse des colonnes
                columns = []
                for col in self.inspector.get_columns(view_name, schema):
                    column_info = {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'default': str(col.get('default', None))
                    }

                    # Statistiques basiques uniquement
                    try:
                        stats = self._get_column_statistics(schema, view_name, col['name'], str(col['type']))
                        if stats:
                            column_info['statistics'] = stats
                    except Exception as e:
                        logger.warning(f"Error getting statistics for column {col['name']}: {str(e)}")

                    columns.append(column_info)

                view_analysis['columns'] = columns

                # 3. Statistiques globales simplifiées
                try:
                    stats_query = text(f"SELECT COUNT(*) as total_rows FROM {schema}.{view_name}")
                    total_rows = connection.execute(stats_query).scalar()
                    view_analysis['statistics']['total_rows'] = total_rows
                except Exception as e:
                    logger.warning(f"Error getting view row count: {str(e)}")

                # 4. Analyse de complexité basique
                if view_def:
                    view_analysis['complexity'] = {
                        'joins': view_def.upper().count('JOIN'),
                        'conditions': view_def.upper().count('WHERE'),
                        'aggregations': view_def.upper().count('GROUP BY')
                    }

            return view_analysis

        except Exception as e:
            logger.error(f"Error analyzing view {schema}.{view_name}: {str(e)}")
            raise

    def _get_basic_metadata(self, schema: str, table_name: str) -> Dict[str, Any]:
        """Récupère les métadonnées de base d'une table."""
        try:
            with self.engine.connect() as connection:
                # Informations de base
                info_query = text("""
                    SELECT 
                        table_catalog,
                        table_schema,
                        table_type,
                        is_insertable_into,
                        commit_action
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                    AND table_name = :table
                """)

                basic_info = dict(connection.execute(
                    info_query,
                    {"schema": schema, "table": table}
                ).fetchone())

                # Taille et statistiques de stockage
                if self.engine.dialect.name == 'postgresql':
                    size_query = text("""
                        SELECT
                            pg_size_pretty(pg_total_relation_size(:table)) as total_size,
                            pg_size_pretty(pg_table_size(:table)) as table_size,
                            pg_size_pretty(pg_indexes_size(:table)) as index_size
                    """)
                    size_info = dict(connection.execute(
                        size_query,
                        {"table": f"{schema}.{table_name}"}
                    ).fetchone())
                    basic_info.update(size_info)

                return basic_info
        except Exception as e:
            logger.error(f"Error getting basic metadata: {str(e)}")
            return {}

    def _analyze_columns(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """Analyse détaillée des colonnes."""
        try:
            columns = []
            for col in self.inspector.get_columns(table_name, schema):
                column_info = {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default', None)),
                    'statistics': self._get_column_statistics(schema, table_name, col['name'], str(col['type'])),
                    'quality_metrics': self._get_column_quality_metrics(schema, table_name, col['name'],
                                                                        str(col['type'])),
                    'distribution': self._get_column_distribution(schema, table_name, col['name'], str(col['type']))
                }
                columns.append(column_info)
            return columns
        except Exception as e:
            logger.error(f"Error analyzing columns: {str(e)}")
            return []

    def _get_column_statistics(self, schema: str, table_name: str, column_name: str, column_type: str) -> Dict[
        str, Any]:
        """
        Calcule les statistiques complètes pour une colonne.

        Args:
            schema: Nom du schéma
            table_name: Nom de la table
            column_name: Nom de la colonne
            column_type: Type de la colonne

        Returns:
            Dict contenant toutes les statistiques calculées
        """
        try:
            stats = {}

            with self.engine.connect() as connection:
                # 1. Statistiques de base pour tous les types
                base_query = text(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT(DISTINCT {column_name}) as unique_count,
                        COUNT(*) - COUNT({column_name}) as null_count
                    FROM {schema}.{table_name}
                """)

                result = connection.execute(base_query).fetchone()
                if result:
                    stats.update({
                        'total_count': result[0],
                        'unique_count': result[1],
                        'null_count': result[2]
                    })

                # 2. Statistiques pour types numériques
                if any(t in column_type.lower() for t in ['int', 'float', 'numeric', 'decimal', 'real', 'double']):
                    num_query = text(f"""
                        WITH stats AS (
                            SELECT
                                MIN({column_name}::numeric) as min_val,
                                MAX({column_name}::numeric) as max_val,
                                AVG({column_name}::numeric) as mean_val,
                                STDDEV({column_name}::numeric) as std_dev,
                                VARIANCE({column_name}::numeric) as variance,
                                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column_name}::numeric) as q1,
                                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}::numeric) as median,
                                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column_name}::numeric) as q3
                            FROM {schema}.{table_name}
                            WHERE {column_name} IS NOT NULL
                        )
                        SELECT * FROM stats
                    """)

                    num_result = connection.execute(num_query).fetchone()
                    if num_result:
                        stats['numerical'] = {
                            'min': self._safe_float_conversion(num_result.min_val),
                            'max': self._safe_float_conversion(num_result.max_val),
                            'mean': self._safe_float_conversion(num_result.mean_val),
                            'std_dev': self._safe_float_conversion(num_result.std_dev),
                            'variance': self._safe_float_conversion(num_result.variance),
                            'q1': self._safe_float_conversion(num_result.q1),
                            'median': self._safe_float_conversion(num_result.median),
                            'q3': self._safe_float_conversion(num_result.q3)
                        }

                        # Distribution pour les valeurs numériques avec gestion min=max
                        if stats['numerical']['min'] is not None and stats['numerical']['max'] is not None:
                            hist_query = text(f"""
                                WITH bounds AS (
                                    SELECT 
                                        MIN({column_name}::numeric) as min_val,
                                        MAX({column_name}::numeric) as max_val,
                                        CASE 
                                            WHEN MIN({column_name}::numeric) = MAX({column_name}::numeric) 
                                            THEN 1.0 
                                            ELSE (MAX({column_name}::numeric) - MIN({column_name}::numeric)) / 10.0 
                                        END as bucket_size
                                    FROM {schema}.{table_name}
                                    WHERE {column_name} IS NOT NULL
                                )
                                SELECT 
                                    CASE 
                                        WHEN min_val = max_val THEN 1
                                        ELSE width_bucket({column_name}::numeric, 
                                            min_val, 
                                            CASE 
                                                WHEN min_val = max_val THEN min_val + 0.1
                                                ELSE max_val 
                                            END, 
                                            10)
                                    END as bucket,
                                    COUNT(*) as frequency,
                                    MIN({column_name}::numeric) as bucket_min,
                                    MAX({column_name}::numeric) as bucket_max,
                                    AVG({column_name}::numeric) as bucket_avg
                                FROM {schema}.{table_name}, bounds
                                WHERE {column_name} IS NOT NULL
                                GROUP BY bucket
                                ORDER BY bucket
                            """)

                            buckets = []
                            for row in connection.execute(hist_query):
                                buckets.append({
                                    'bucket': row.bucket,
                                    'frequency': row.frequency,
                                    'min_value': self._safe_float_conversion(row.bucket_min),
                                    'max_value': self._safe_float_conversion(row.bucket_max),
                                    'avg_value': self._safe_float_conversion(row.bucket_avg)
                                })

                            if buckets:
                                stats['distribution'] = buckets

                # 3. Statistiques pour types non numériques
                else:
                    # 3.1 Échantillons de valeurs distinctes
                    if 'date' in column_type.lower() or 'timestamp' in column_type.lower():
                        # Pour les dates/timestamps
                        sample_query = text(f"""
                            WITH DateRanks AS (
                                SELECT DISTINCT {column_name},
                                       ntile(5) OVER (ORDER BY {column_name}) as bucket
                                FROM {schema}.{table_name}
                                WHERE {column_name} IS NOT NULL
                            )
                            SELECT {column_name}
                            FROM DateRanks
                            WHERE bucket IN (1, 3, 5)
                            ORDER BY {column_name}
                            LIMIT 5
                        """)

                        # Statistiques supplémentaires pour les dates
                        date_stats_query = text(f"""
                            SELECT 
                                MIN({column_name}) as min_date,
                                MAX({column_name}) as max_date,
                                COUNT(DISTINCT DATE_TRUNC('year', {column_name})) as distinct_years,
                                COUNT(DISTINCT DATE_TRUNC('month', {column_name})) as distinct_months,
                                COUNT(DISTINCT DATE_TRUNC('day', {column_name})) as distinct_days
                            FROM {schema}.{table_name}
                            WHERE {column_name} IS NOT NULL
                        """)

                        date_stats = connection.execute(date_stats_query).fetchone()
                        if date_stats:
                            stats['date_stats'] = {
                                'min_date': date_stats.min_date.isoformat() if date_stats.min_date else None,
                                'max_date': date_stats.max_date.isoformat() if date_stats.max_date else None,
                                'distinct_years': date_stats.distinct_years,
                                'distinct_months': date_stats.distinct_months,
                                'distinct_days': date_stats.distinct_days
                            }

                    else:
                        # Pour les chaînes de caractères et autres types
                        sample_query = text(f"""
                                WITH DistinctValues AS (
                                    SELECT DISTINCT {column_name},
                                           length({column_name}::text) as str_length,
                                           COUNT(*) OVER (PARTITION BY {column_name}) as frequency
                                    FROM {schema}.{table_name}
                                    WHERE {column_name} IS NOT NULL
                                ),
                                BucketedValues AS (
                                    SELECT *,
                                           width_bucket(str_length, 
                                                      MIN(str_length) OVER (), 
                                                      GREATEST(MAX(str_length) OVER (), MIN(str_length) OVER () + 1),
                                                      5) as bucket
                                    FROM DistinctValues
                                )
                                SELECT {column_name}
                                FROM (
                                    SELECT *,
                                           ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY frequency DESC) as rn
                                    FROM BucketedValues
                                ) ranked
                                WHERE rn = 1
                                ORDER BY bucket
                                LIMIT 5
                            """)

                        # Statistiques textuelles adaptées pour tous les types
                        text_stats_query = text(f"""
                            SELECT 
                                AVG(length({column_name}::text)) as avg_length,
                                MIN(length({column_name}::text)) as min_length,
                                MAX(length({column_name}::text)) as max_length,
                                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY length({column_name}::text)) as median_length,
                                COUNT(DISTINCT length({column_name}::text)) as distinct_lengths,
                                COUNT(DISTINCT lower({column_name}::text)) as distinct_case_insensitive
                            FROM {schema}.{table_name}
                            WHERE {column_name} IS NOT NULL
                        """)

                        text_stats = connection.execute(text_stats_query).fetchone()
                        if text_stats:
                            stats['text_stats'] = {
                                'avg_length': float(text_stats.avg_length) if text_stats.avg_length else None,
                                'min_length': text_stats.min_length,
                                'max_length': text_stats.max_length,
                                'median_length': float(text_stats.median_length) if text_stats.median_length else None,
                                'distinct_lengths': text_stats.distinct_lengths,
                                'distinct_case_insensitive': text_stats.distinct_case_insensitive
                            }

                    # Collecte des échantillons
                    try:
                        sample_results = connection.execute(sample_query).fetchall()
                        if sample_results:
                            if 'date' in column_type.lower() or 'timestamp' in column_type.lower():
                                stats['sample_values'] = [
                                    row[0].isoformat() if row[0] else None
                                    for row in sample_results
                                ]
                            else:
                                stats['sample_values'] = [str(row[0]) for row in sample_results]
                    except Exception as e:
                        logger.warning(f"Error getting sample values for {column_name}: {str(e)}")

                return stats

        except Exception as e:
            logger.error(f"Error getting column statistics for {schema}.{table_name}.{column_name}: {str(e)}")
            return {
                'total_count': 0,
                'unique_count': 0,
                'null_count': 0,
                'error': str(e)
            }

    def start_analysis(self, schemas: List[str], selected_objects: Dict, exclude_pattern: str = None):
        """Lance l'analyse en deux étapes."""
        try:
            analysis_results = {}
            llm_analyzer = LLMAnalyzer()

            for schema in schemas:
                schema_objects = selected_objects.get(schema, {})

                analysis_results[schema] = {
                    'tables': {},
                    'views': {},
                    'metadata': {
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                }

                # Analyse des tables et vues
                for obj_type in ['tables', 'views']:
                    for obj_name in schema_objects.get(obj_type, []):
                        try:
                            # Étape 1: Analyse classique
                            obj_analysis = self.analyze_table(schema, obj_name) if obj_type == 'tables' \
                                else self.analyze_view(schema, obj_name)

                            # Étape 2: Analyse LLM
                            llm_results = llm_analyzer.analyze_database_object(
                                object_data=obj_analysis,
                                object_type=obj_type[:-1]  # 'table' ou 'view'
                            )

                            # Fusion des résultats
                            if llm_results and 'analysis_results' in llm_results:
                                obj_analysis['llm_analysis'] = llm_results['analysis_results']

                            analysis_results[schema][obj_type][obj_name] = obj_analysis

                        except Exception as e:
                            logger.error(f"Error analyzing {schema}.{obj_name}: {str(e)}")
                            analysis_results[schema][obj_type][obj_name] = {
                                'error': str(e),
                                'metadata': {'analysis_timestamp': datetime.now().isoformat()}
                            }

            return analysis_results

        except Exception as e:
            logger.error(f"Error during analysis process: {str(e)}")
            raise

    def _get_column_quality_metrics(self, schema: str, table_name: str, column_name: str, column_type: str) -> Dict[
        str, Any]:
        """Analyse la qualité des données d'une colonne."""
        try:
            with self.engine.connect() as connection:
                metrics = {}

                # Détection des valeurs vides pour les chaînes
                if 'char' in column_type.lower():
                    empty_query = text(f"""
                        SELECT 
                            COUNT(*) FILTER (WHERE {column_name} = '') as empty_count,
                            COUNT(*) FILTER (WHERE {column_name} ~ '^[[:space:]]*$') as blank_count
                        FROM {schema}.{table_name}
                        WHERE {column_name} IS NOT NULL
                    """)
                    metrics.update(dict(connection.execute(empty_query).fetchone()))

                # Validation des dates
                if 'date' in column_type.lower():
                    date_query = text(f"""
                        SELECT 
                            COUNT(*) FILTER (WHERE {column_name} < '1900-01-01') as old_dates,
                            COUNT(*) FILTER (WHERE {column_name} > CURRENT_DATE) as future_dates
                        FROM {schema}.{table_name}
                        WHERE {column_name} IS NOT NULL
                    """)
                    metrics.update(dict(connection.execute(date_query).fetchone()))

                # Détection des valeurs aberrantes pour les nombres
                if any(t in column_type.lower() for t in ['int', 'float', 'numeric', 'decimal']):
                    outlier_query = text(f"""
                        WITH stats AS (
                            SELECT
                                percentile_cont(0.25) WITHIN GROUP (ORDER BY {column_name}::numeric) as q1,
                                percentile_cont(0.75) WITHIN GROUP (ORDER BY {column_name}::numeric) as q3
                            FROM {schema}.{table_name}
                            WHERE {column_name} IS NOT NULL
                        )
                        SELECT 
                            COUNT(*) FILTER (
                                WHERE {column_name}::numeric < q1 - 1.5 * (q3 - q1)
                                OR {column_name}::numeric > q3 + 1.5 * (q3 - q1)
                            ) as outlier_count
                        FROM {schema}.{table_name}, stats
                        WHERE {column_name} IS NOT NULL
                    """)
                    metrics.update(dict(connection.execute(outlier_query).fetchone()))

                return metrics
        except Exception as e:
            logger.error(f"Error getting column quality metrics: {str(e)}")
            return {}

    def _get_column_distribution(self, schema: str, table_name: str, column_name: str, column_type: str) -> Dict[
        str, Any]:
        """Analyse la distribution des données d'une colonne."""
        try:
            with self.engine.connect() as connection:
                distribution = {}

                # Distribution pour les colonnes numériques
                if any(t in column_type.lower() for t in ['int', 'float', 'numeric', 'decimal']):
                    # Création des buckets pour l'histogramme
                    hist_query = text(f"""
                        WITH bounds AS (
                            SELECT 
                                MIN({column_name}::numeric) as min_val,
                                MAX({column_name}::numeric) as max_val
                            FROM {schema}.{table_name}
                            WHERE {column_name} IS NOT NULL
                        )
                        SELECT 
                            width_bucket({column_name}::numeric, min_val, max_val, 9) as bucket,
                            COUNT(*) as frequency,
                            MIN({column_name}::numeric) as bucket_min,
                            MAX({column_name}::numeric) as bucket_max
                        FROM {schema}.{table_name}, bounds
                        WHERE {column_name} IS NOT NULL
                        GROUP BY bucket
                        ORDER BY bucket
                    """)
                    distribution['histogram'] = [dict(row) for row in connection.execute(hist_query)]

                # Distribution pour les colonnes catégorielles
                else:
                    cat_query = text(f"""
                        SELECT 
                            {column_name} as value,
                            COUNT(*) as frequency,
                            COUNT(*)::float / SUM(COUNT(*)) OVER () as percentage
                        FROM {schema}.{table_name}
                        WHERE {column_name} IS NOT NULL
                        GROUP BY {column_name}
                        ORDER BY COUNT(*) DESC
                        LIMIT 20
                    """)
                    distribution['categories'] = [dict(row) for row in connection.execute(cat_query)]

                return distribution
        except Exception as e:
            logger.error(f"Error getting column distribution: {str(e)}")
            return {}

    def _get_primary_keys(self, schema: str, table_name: str) -> Dict[str, Any]:
        """Récupère les informations sur les clés primaires."""
        try:
            pk_info = self.inspector.get_pk_constraint(table_name, schema)

            if pk_info:
                # Ajout d'informations supplémentaires sur la clé primaire
                with self.engine.connect() as connection:
                    pk_columns = pk_info['constrained_columns']
                    pk_stats_query = text(f"""
                        SELECT 
                            COUNT(*) as total_rows,
                            COUNT(DISTINCT ({','.join(pk_columns)})) as unique_values
                        FROM {schema}.{table_name}
                    """)
                    pk_stats = dict(connection.execute(pk_stats_query).fetchone())
                    pk_info['statistics'] = pk_stats

            return pk_info or {}
        except Exception as e:
            logger.error(f"Error getting primary keys: {str(e)}")
            return {}

    def _get_foreign_keys(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Récupère les clés étrangères avec gestion sûre des conversions.
        """
        try:
            fks = self.inspector.get_foreign_keys(table_name, schema)

            # Enrichissement des informations
            for fk in fks:
                with self.engine.connect() as connection:
                    fk_stats_query = text(f"""
                        SELECT 
                            COUNT(DISTINCT {fk['constrained_columns'][0]}) as unique_values,
                            COUNT(*) - COUNT({fk['constrained_columns'][0]}) as null_count,
                            COUNT(*) as total_rows
                        FROM {schema}.{table_name}
                    """)

                    result = connection.execute(fk_stats_query).fetchone()
                    if result:
                        fk['statistics'] = {
                            'unique_values': result[0],
                            'null_count': result[1],
                            'total_rows': result[2]
                        }

            return fks

        except Exception as e:
            logger.error(f"Error getting foreign keys for {schema}.{table_name}: {str(e)}")
            return []

        except Exception as e:
            logger.error(f"Error getting foreign keys for {schema}.{table_name}: {str(e)}")
            return []

    def _safe_float_conversion(self, value: Any) -> Optional[float]:
        """
        Convertit de manière sûre une valeur en float.

        Args:
            value: Valeur à convertir

        Returns:
            float ou None si la conversion échoue
        """
        if value is None:
            return None
        try:
            if isinstance(value, Decimal):
                return float(value)
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_indexes(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Récupère les informations sur les index de manière optimisée.

        Args:
            schema: Nom du schéma
            table_name: Nom de la table

        Returns:
            Liste des index avec leurs statistiques
        """
        try:
            indexes = self.inspector.get_indexes(table_name, schema)

            # Enrichissement avec les statistiques
            with self.engine.connect() as connection:
                for idx in indexes:
                    try:
                        # Requête corrigée pour les statistiques d'index
                        stats_query = text("""
                            SELECT 
                                pg_size_pretty(pg_relation_size(i.indexrelid)) as index_size,
                                s.idx_scan,
                                s.idx_tup_read,
                                s.idx_tup_fetch
                            FROM pg_class t
                            JOIN pg_index ix ON t.oid = ix.indrelid
                            JOIN pg_class i ON ix.indexrelid = i.oid
                            JOIN pg_namespace n ON n.oid = i.relnamespace
                            LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid
                            WHERE n.nspname = :schema 
                            AND t.relname = :table
                            AND i.relname = :index_name
                        """)

                        result = connection.execute(
                            stats_query,
                            {
                                "schema": schema,
                                "table": table_name,
                                "index_name": idx['name']
                            }
                        ).fetchone()

                        if result:
                            idx['statistics'] = {
                                'size': result[0],
                                'scans': result[1],
                                'tuples_read': result[2],
                                'tuples_fetched': result[3]
                            }

                            # Calcul de la sélectivité
                            if idx['column_names']:
                                try:
                                    sel_query = text(f"""
                                        SELECT 
                                            NULLIF(COUNT(DISTINCT ({','.join(idx['column_names'])})), 0)::float / 
                                            NULLIF(COUNT(*), 0)::float as selectivity
                                        FROM {schema}.{table_name}
                                    """)

                                    selectivity = connection.execute(sel_query).scalar()
                                    if selectivity is not None:
                                        idx['selectivity'] = float(selectivity)
                                except Exception as e:
                                    logger.warning(f"Error calculating selectivity for index {idx['name']}: {str(e)}")

                    except Exception as e:
                        logger.warning(f"Error getting statistics for index {idx['name']}: {str(e)}")
                        idx['statistics'] = None

            return indexes

        except Exception as e:
            logger.error(f"Error getting indexes: {str(e)}")
            return []

    def _get_constraints(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Récupère les contraintes avec gestion sûre des résultats.
        """
        try:
            with self.engine.connect() as connection:
                constraints_query = text("""
                    SELECT 
                        constraint_name,
                        constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_schema = :schema 
                    AND table_name = :table_name
                """)

                results = connection.execute(
                    constraints_query,
                    {"schema": schema, "table_name": table_name}
                ).fetchall()

                return [
                    {
                        'name': row[0],
                        'type': row[1]
                    }
                    for row in results
                ]

        except Exception as e:
            logger.error(f"Error getting constraints: {str(e)}")
            return []

    def _analyze_data_quality(self, schema: str, table_name: str) -> Dict[str, Any]:
        """
        Analyse approfondie de la qualité des données.

        Args:
            schema: Nom du schéma
            table_name: Nom de la table

        Returns:
            Dict contenant les métriques de qualité
        """
        try:
            quality_metrics = {
                'completeness': {},
                'validity': {},
                'consistency': {},
                'patterns': {}
            }

            with self.engine.connect() as connection:
                # Analyse de la complétude (valeurs nulles et vides)
                for col in self.inspector.get_columns(table_name, schema):
                    col_name = col['name']
                    completeness_query = text(f"""
                        SELECT 
                            COUNT(*) as total_rows,
                            COUNT({col_name}) as non_null_rows,
                            CASE WHEN pg_typeof({col_name}) = 'text'::regtype
                                THEN COUNT(*) FILTER (WHERE {col_name} = '')
                                ELSE 0
                            END as empty_strings
                        FROM {schema}.{table_name}
                    """)
                    stats = dict(connection.execute(completeness_query).fetchone())
                    quality_metrics['completeness'][col_name] = {
                        'null_percentage': (stats['total_rows'] - stats['non_null_rows']) / stats['total_rows'] * 100,
                        'empty_percentage': stats['empty_strings'] / stats['total_rows'] * 100 if stats[
                                                                                                      'total_rows'] > 0 else 0
                    }

                # Analyse de la validité (formats de données)
                for col in self.inspector.get_columns(table_name, schema):
                    col_name = col['name']
                    col_type = str(col['type'])

                    if 'date' in col_type.lower():
                        # Vérification des dates
                        date_query = text(f"""
                            SELECT 
                                COUNT(*) FILTER (WHERE {col_name}::text ~ '^\d{4}-\d{2}-\d{2}$') as valid_dates,
                                COUNT(*) FILTER (WHERE {col_name} < '1900-01-01') as old_dates,
                                COUNT(*) FILTER (WHERE {col_name} > CURRENT_DATE) as future_dates
                            FROM {schema}.{table_name}
                            WHERE {col_name} IS NOT NULL
                        """)
                        quality_metrics['validity'][col_name] = dict(connection.execute(date_query).fetchone())

                    elif 'numeric' in col_type.lower():
                        # Vérification des nombres
                        number_query = text(f"""
                            WITH stats AS (
                                SELECT
                                    percentile_cont(0.25) WITHIN GROUP (ORDER BY {col_name}::numeric) as q1,
                                    percentile_cont(0.75) WITHIN GROUP (ORDER BY {col_name}::numeric) as q3
                                FROM {schema}.{table_name}
                                WHERE {col_name} IS NOT NULL
                            )
                            SELECT 
                                COUNT(*) as total_values,
                                COUNT(*) FILTER (
                                    WHERE {col_name}::numeric < q1 - 1.5 * (q3 - q1)
                                    OR {col_name}::numeric > q3 + 1.5 * (q3 - q1)
                                ) as outliers
                            FROM {schema}.{table_name}, stats
                            WHERE {col_name} IS NOT NULL
                        """)
                        quality_metrics['validity'][col_name] = dict(connection.execute(number_query).fetchone())

                # Analyse de la consistance (patterns et formats)
                for col in self.inspector.get_columns(table_name, schema):
                    if 'char' in str(col['type']).lower():
                        pattern_query = text(f"""
                            SELECT 
                                COUNT(*) FILTER (WHERE {col['name']} ~ '^\d+$') as numeric_pattern,
                                COUNT(*) FILTER (WHERE {col['name']} ~ '^[A-Za-z]+$') as alpha_pattern,
                                COUNT(*) FILTER (WHERE {col['name']} ~ '^[A-Za-z0-9]+$') as alphanumeric_pattern
                            FROM {schema}.{table_name}
                            WHERE {col['name']} IS NOT NULL
                        """)
                        quality_metrics['patterns'][col['name']] = dict(connection.execute(pattern_query).fetchone())

            return quality_metrics
        except Exception as e:
            logger.error(f"Error analyzing data quality: {str(e)}")
            return {}

    def _detect_potential_relations(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """
        Détecte les relations potentielles basées sur les conventions de nommage et les données.

        Args:
            schema: Nom du schéma
            table_name: Nom de la table

        Returns:
            Liste des relations potentielles
        """
        try:
            potential_relations = []

            # Patterns communs pour les clés étrangères
            fk_patterns = [
                '_id$', 'id_', '_code$', 'code_',
                '_key$', 'key_', '_ref$', 'ref_'
            ]

            # Obtention de toutes les tables du schéma
            all_tables = self.inspector.get_table_names(schema)

            with self.engine.connect() as connection:
                for col in self.inspector.get_columns(table_name, schema):
                    col_name = col['name'].lower()

                    # Vérification des patterns de nommage
                    for pattern in fk_patterns:
                        import re
                        if re.search(pattern, col_name):
                            # Extraction du nom de base
                            base_name = re.sub(pattern, '', col_name)

                            # Recherche de tables correspondantes
                            for other_table in all_tables:
                                if base_name in other_table.lower():
                                    # Analyse de correspondance des valeurs
                                    value_match_query = text(f"""
                                        SELECT 
                                            COUNT(DISTINCT a.{col_name}) as matching_values,
                                            COUNT(DISTINCT b.id) as total_values
                                        FROM {schema}.{table_name} a
                                        LEFT JOIN {schema}.{other_table} b
                                        ON a.{col_name} = b.id
                                    """)

                                    try:
                                        stats = dict(connection.execute(value_match_query).fetchone())
                                        if stats['matching_values'] > 0:
                                            match_percentage = (stats['matching_values'] / stats['total_values'] * 100
                                                                if stats['total_values'] > 0 else 0)

                                            if match_percentage > 50:  # Seuil de confiance
                                                potential_relations.append({
                                                    'from_table': table_name,
                                                    'from_column': col_name,
                                                    'to_table': other_table,
                                                    'to_column': 'id',
                                                    'match_percentage': match_percentage,
                                                    'pattern_matched': pattern,
                                                    'statistics': stats
                                                })
                                    except Exception as e:
                                        logger.warning(f"Error checking value match for {col_name}: {str(e)}")
                                        continue

            return potential_relations
        except Exception as e:
            logger.error(f"Error detecting potential relations: {str(e)}")
            return []

    def _get_table_statistics(self, schema: str, table_name: str) -> Dict[str, Any]:
        """
        Calcule des statistiques globales sur la table.

        Args:
            schema: Nom du schéma
            table_name: Nom de la table

        Returns:
            Dict contenant les statistiques globales
        """
        try:
            with self.engine.connect() as connection:
                stats = {}

                # Statistiques de base
                base_query = text(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        pg_size_pretty(pg_total_relation_size('{schema}.{table_name}')) as total_size,
                        pg_size_pretty(pg_table_size('{schema}.{table_name}')) as table_size,
                        pg_size_pretty(pg_indexes_size('{schema}.{table_name}')) as index_size
                    FROM {schema}.{table_name}
                """)
                stats.update(dict(connection.execute(base_query).fetchone()))

                # Statistiques de modification
                if self.engine.dialect.name == 'postgresql':
                    mod_query = text("""
                        SELECT 
                            n_live_tup as live_rows,
                            n_dead_tup as dead_rows,
                            n_mod_since_analyze as modifications,
                            last_vacuum,
                            last_autovacuum,
                            last_analyze,
                            last_autoanalyze
                        FROM pg_stat_user_tables
                        WHERE schemaname = :schema
                        AND relname = :table
                    """)

                    mod_stats = dict(connection.execute(
                        mod_query,
                        {"schema": schema, "table": table_name}
                    ).fetchone() or {})

                    stats['modification_stats'] = mod_stats

                return stats
        except Exception as e:
            logger.error(f"Error getting table statistics: {str(e)}")
            return {}


def _get_column_sample(self, schema: str, table_name: str, column_name: str, sample_size: int = 10000) -> List[Any]:
    """
    Extrait un échantillon aléatoire des valeurs d'une colonne.

    Args:
        schema: Nom du schéma
        table_name: Nom de la table
        column_name: Nom de la colonne
        sample_size: Taille de l'échantillon

    Returns:
        Liste des valeurs échantillonnées
    """
    try:
        with self.engine.connect() as connection:
            # Utilise TABLESAMPLE pour PostgreSQL ou un RANDOM pour d'autres
            if self.engine.dialect.name == 'postgresql':
                query = text(f"""
                    SELECT {column_name}
                    FROM {schema}.{table_name} TABLESAMPLE SYSTEM(10)
                    WHERE {column_name} IS NOT NULL
                    LIMIT :sample_size
                """)
            else:
                query = text(f"""
                    SELECT {column_name}
                    FROM {schema}.{table_name}
                    WHERE {column_name} IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT :sample_size
                """)

            result = connection.execute(query, {"sample_size": sample_size}).fetchall()
            return [row[0] for row in result]

    except Exception as e:
        logger.error(f"Error getting column sample: {str(e)}")
        return []