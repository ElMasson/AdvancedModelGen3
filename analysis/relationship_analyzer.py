# analysis/relationship_analyzer.py
from typing import Dict, List, Optional, Any, Union
import logging
import json
from datetime import datetime
from core.config import Config
from core.types import TableInfo, AnalysisResult
from analysis.llm_analyzer import LLMAnalyzer
import networkx as nx
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class RelationshipAnalyzer:
    """Analyseur de relations entre les tables."""

    def __init__(self, llm_analyzer: Optional[LLMAnalyzer] = None):
        """
        Initialise l'analyseur de relations.

        Args:
            llm_analyzer: Instance optionnelle de LLMAnalyzer
        """
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()
        self.graph = nx.DiGraph()

    def analyze_schema_relationships(self, schema_metadata: Dict) -> Dict:
        """
        Analyse complète des relations du schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant toutes les relations analysées
        """
        try:
            # Extraction des relations explicites
            explicit_relations = self._extract_explicit_relations(schema_metadata)

            # Détection des relations potentielles
            potential_relations = self._detect_potential_relations(schema_metadata)

            # Validation des relations
            validated_relations = self._validate_relationships(schema_metadata, potential_relations)

            # Enrichissement via LLM
            enriched_relations = self._enrich_relationships_with_llm(
                schema_metadata,
                explicit_relations,
                validated_relations
            )

            return {
                'explicit_relations': explicit_relations,
                'potential_relations': validated_relations,
                'enriched_relations': enriched_relations,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }

        except Exception as e:
            logger.error(f"Error in schema relationship analysis: {str(e)}")
            return {'error': str(e)}

    def _extract_explicit_relations(self, schema_metadata: Dict) -> List[Dict]:
        """
        Extrait les relations explicites (clés étrangères, etc.).

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Liste des relations explicites
        """
        relations = []

        try:
            for table_name, table_data in schema_metadata.get('tables', {}).items():
                # Relations via clés étrangères
                for fk in table_data.get('foreign_keys', []):
                    relations.append({
                        'source': table_name,
                        'target': fk['referred_table'],
                        'type': 'foreign_key',
                        'columns': {
                            'source': fk['constrained_columns'],
                            'target': fk['referred_columns']
                        },
                        'confidence': 1.0,
                        'metadata': {
                            'constraint_name': fk.get('name'),
                            'ondelete': fk.get('ondelete'),
                            'onupdate': fk.get('onupdate')
                        }
                    })

                # Relations via indexes
                for idx in table_data.get('indexes', []):
                    if len(idx.get('columns', [])) > 1:  # Index multi-colonnes
                        relations.append({
                            'source': table_name,
                            'type': 'index_relationship',
                            'columns': idx['columns'],
                            'confidence': 0.8,
                            'metadata': {
                                'index_name': idx.get('name'),
                                'unique': idx.get('unique', False)
                            }
                        })

            return relations

        except Exception as e:
            logger.error(f"Error extracting explicit relations: {str(e)}")
            return []

    def _detect_potential_relations(self, schema_metadata: Dict) -> List[Dict]:
        """
        Détecte les relations potentielles basées sur les patterns.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Liste des relations potentielles
        """
        try:
            potential_relations = []

            # Patterns de nommage courants pour les clés étrangères
            naming_patterns = [
                ('_id$', 0.8),
                ('id_', 0.8),
                ('_code$', 0.7),
                ('code_', 0.7),
                ('_key$', 0.6),
                ('key_', 0.6),
                ('_ref$', 0.6),
                ('ref_', 0.6)
            ]

            tables = schema_metadata.get('tables', {})

            for source_table, source_data in tables.items():
                for column in source_data.get('columns', []):
                    col_name = column['name'].lower()

                    # Vérification des patterns de nommage
                    for pattern, confidence in naming_patterns:
                        if re.search(pattern, col_name):
                            # Extraction du nom de base
                            base_name = re.sub(pattern, '', col_name)

                            # Recherche de tables correspondantes
                            matches = [
                                table for table in tables.keys()
                                if base_name in table.lower()
                            ]

                            for target_table in matches:
                                potential_relations.append({
                                    'source': source_table,
                                    'target': target_table,
                                    'type': 'naming_pattern',
                                    'columns': {
                                        'source': [col_name],
                                        'target': ['id']
                                    },
                                    'confidence': confidence,
                                    'metadata': {
                                        'pattern_matched': pattern,
                                        'base_name': base_name
                                    }
                                })

            # Analyse des similarités de données
            for source_table, source_data in tables.items():
                for source_col in source_data.get('columns', []):
                    # Ne considérer que les colonnes numériques ou de type ID
                    if self._is_potential_key_column(source_col):
                        for target_table, target_data in tables.items():
                            if target_table != source_table:
                                for target_col in target_data.get('columns', []):
                                    if self._columns_potentially_related(source_col, target_col):
                                        potential_relations.append({
                                            'source': source_table,
                                            'target': target_table,
                                            'type': 'data_similarity',
                                            'columns': {
                                                'source': [source_col['name']],
                                                'target': [target_col['name']]
                                            },
                                            'confidence': 0.6,
                                            'metadata': {
                                                'source_type': str(source_col['type']),
                                                'target_type': str(target_col['type'])
                                            }
                                        })

            return potential_relations

        except Exception as e:
            logger.error(f"Error detecting potential relations: {str(e)}")
            return []

    def _validate_relationships(self, schema_metadata: Dict, potential_relations: List[Dict]) -> List[Dict]:
        """
        Valide les relations potentielles détectées.

        Args:
            schema_metadata: Métadonnées du schéma
            potential_relations: Liste des relations potentielles

        Returns:
            Liste des relations validées
        """
        try:
            validated_relations = []

            for relation in potential_relations:
                validation_score = self._calculate_validation_score(
                    schema_metadata,
                    relation
                )

                if validation_score >= 0.5:  # Seuil minimum de confiance
                    validated_relation = {
                        **relation,
                        'confidence': relation['confidence'] * validation_score,
                        'validation': {
                            'score': validation_score,
                            'timestamp': datetime.now().isoformat(),
                            'checks_passed': self._get_validation_details(
                                schema_metadata,
                                relation
                            )
                        }
                    }
                    validated_relations.append(validated_relation)

            return validated_relations

        except Exception as e:
            logger.error(f"Error validating relationships: {str(e)}")
            return []

    def _enrich_relationships_with_llm(self, schema_metadata: Dict, explicit_relations: List[Dict],
                                       validated_relations: List[Dict]) -> List[Dict]:
        """
        Enrichit les relations avec l'analyse LLM.
        """
        try:
            # Création d'un prompt plus structuré
            prompt = f"""
            Analyze these database relationships and provide insights in JSON format.

            Schema context:
            {json.dumps(schema_metadata, indent=2)}

            Explicit relationships:
            {json.dumps(explicit_relations, indent=2)}

            Potential relationships:
            {json.dumps(validated_relations, indent=2)}

            For each relationship analyze:
            1. Business purpose
            2. Data flow implications
            3. Integrity requirements
            4. Usage patterns
            5. Performance considerations
            
            Answer in French
            """

            llm_analysis = self.llm_analyzer.analyze_with_prompt(prompt)

            # Enrichissement des relations avec l'analyse LLM
            enriched_relations = []
            all_relations = explicit_relations + validated_relations

            for relation in all_relations:
                enriched_relation = {
                    **relation,
                    'llm_analysis': self._match_llm_analysis(relation, llm_analysis)
                }
                enriched_relations.append(enriched_relation)

            return enriched_relations

        except Exception as e:
            logger.error(f"Error enriching relationships with LLM: {str(e)}")
            return explicit_relations + validated_relations  # Retourner les relations non enrichies en cas d'erreur

    def _is_potential_key_column(self, column: Dict) -> bool:
        """
        Vérifie si une colonne est potentiellement une clé.

        Args:
            column: Métadonnées de la colonne

        Returns:
            bool indiquant si la colonne est potentiellement une clé
        """
        # Types courants pour les clés
        key_types = {'integer', 'bigint', 'smallint', 'uuid'}

        # Patterns de nommage courants pour les clés
        key_patterns = {'id', 'code', 'key', 'ref', 'num'}

        column_type = str(column.get('type', '')).lower()
        column_name = column.get('name', '').lower()

        # Vérification du type
        is_key_type = any(kt in column_type for kt in key_types)

        # Vérification du nom
        is_key_name = any(kp in column_name for kp in key_patterns)

        return is_key_type or is_key_name

    def _columns_potentially_related(self, source_col: Dict, target_col: Dict) -> bool:
        """
        Vérifie si deux colonnes sont potentiellement reliées.

        Args:
            source_col: Colonne source
            target_col: Colonne cible

        Returns:
            bool indiquant si les colonnes sont potentiellement reliées
        """

        # Vérification des types compatibles
        def get_base_type(col_type: str) -> str:
            col_type = str(col_type).lower()
            if any(num in col_type for num in ['int', 'serial']):
                return 'integer'
            if 'char' in col_type:
                return 'string'
            return col_type

        source_type = get_base_type(source_col.get('type', ''))
        target_type = get_base_type(target_col.get('type', ''))

        if source_type != target_type:
            return False

        # Vérification des noms similaires
        source_name = source_col.get('name', '').lower()
        target_name = target_col.get('name', '').lower()

        # Retrait des préfixes/suffixes courants
        common_affixes = ['id', 'code', 'key', 'ref', 'num']
        for affix in common_affixes:
            source_name = source_name.replace(affix, '')
            target_name = target_name.replace(affix, '')

        # Calcul de similarité
        return self._calculate_name_similarity(source_name, target_name) > 0.6

    def _calculate_validation_score(self, schema_metadata: Dict, relation: Dict) -> float:
        """
        Calcule un score de validation pour une relation.

        Args:
            schema_metadata: Métadonnées du schéma
            relation: Relation à valider

        Returns:
            float représentant le score de validation
        """
        try:
            scores = []

            # Score basé sur les types de données
            scores.append(self._calculate_type_compatibility_score(
                schema_metadata,
                relation
            ))

            # Score basé sur les cardinalités
            scores.append(self._calculate_cardinality_score(
                schema_metadata,
                relation
            ))

            # Score basé sur les patterns de données
            scores.append(self._calculate_data_pattern_score(
                schema_metadata,
                relation
            ))

            # Moyenne pondérée des scores
            weights = [0.4, 0.3, 0.3]  # Importance relative de chaque critère
            return sum(s * w for s, w in zip(scores, weights))

        except Exception as e:
            logger.error(f"Error calculating validation score: {str(e)}")
            return 0.0

    def _get_validation_details(self, schema_metadata: Dict, relation: Dict) -> Dict:
        """
        Récupère les détails de validation d'une relation.

        Args:
            schema_metadata: Métadonnées du schéma
            relation: Relation à valider

        Returns:
            Dict contenant les détails de validation
        """
        return {
            'type_compatibility': self._check_type_compatibility(
                schema_metadata,
                relation
            ),
            'data_patterns': self._check_data_patterns(
                schema_metadata,
                relation
            ),
            'cardinality': self._check_cardinality(
                schema_metadata,
                relation
            ),
            'integrity': self._check_integrity_constraints(
                schema_metadata,
                relation
            )
        }

    def _calculate_type_compatibility_score(self, schema_metadata: Dict, relation: Dict) -> float:
        """
        Calcule le score de compatibilité des types entre les colonnes.

        Args:
            schema_metadata: Métadonnées du schéma
            relation: Relation à évaluer

        Returns:
            float représentant le score de compatibilité
        """
        try:
            tables = schema_metadata.get('tables', {})
            source_table = tables.get(relation['source'], {})
            target_table = tables.get(relation['target'], {})

            source_cols = {col['name']: col['type'] for col in source_table.get('columns', [])}
            target_cols = {col['name']: col['type'] for col in target_table.get('columns', [])}

            source_col_name = relation['columns']['source'][0] if relation.get('columns', {}).get('source') else None
            target_col_name = relation['columns']['target'][0] if relation.get('columns', {}).get('target') else None

            if not (source_col_name and target_col_name):
                return 0.0

            source_type = str(source_cols.get(source_col_name, '')).lower()
            target_type = str(target_cols.get(target_col_name, '')).lower()

            # Types identiques
            if source_type == target_type:
                return 1.0

            # Types numériques compatibles
            numeric_types = {'integer', 'bigint', 'smallint', 'decimal', 'numeric'}
            if any(t in source_type for t in numeric_types) and any(t in target_type for t in numeric_types):
                return 0.8

            # Types texte compatibles
            text_types = {'char', 'varchar', 'text'}
            if any(t in source_type for t in text_types) and any(t in target_type for t in text_types):
                return 0.8

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating type compatibility: {str(e)}")
            return 0.0

    def _calculate_cardinality_score(self, schema_metadata: Dict, relation: Dict) -> float:
        """
        Calcule le score basé sur les cardinalités.

        Args:
            schema_metadata: Métadonnées du schéma
            relation: Relation à évaluer

        Returns:
            float représentant le score de cardinalité
        """
        try:
            tables = schema_metadata.get('tables', {})
            source_table = tables.get(relation['source'], {})
            target_table = tables.get(relation['target'], {})

            source_col_name = relation['columns']['source'][0] if relation.get('columns', {}).get('source') else None
            target_col_name = relation['columns']['target'][0] if relation.get('columns', {}).get('target') else None

            if not (source_col_name and target_col_name):
                return 0.0

            # Analyse des statistiques
            source_stats = next((col['statistics'] for col in source_table.get('columns', [])
                                 if col['name'] == source_col_name), {})
            target_stats = next((col['statistics'] for col in target_table.get('columns', [])
                                 if col['name'] == target_col_name), {})

            # Vérification des valeurs uniques
            source_unique = source_stats.get('unique_count', 0)
            target_unique = target_stats.get('unique_count', 0)

            if source_unique == 0 or target_unique == 0:
                return 0.5  # Score par défaut si les statistiques sont manquantes

            # Calcul du ratio de cardinalité
            ratio = min(source_unique, target_unique) / max(source_unique, target_unique)

            return ratio

        except Exception as e:
            logger.error(f"Error calculating cardinality score: {str(e)}")
            return 0.0

    def _calculate_data_pattern_score(self, schema_metadata: Dict, relation: Dict) -> float:
        """
        Calcule le score basé sur les patterns de données.

        Args:
            schema_metadata: Métadonnées du schéma
            relation: Relation à évaluer

        Returns:
            float représentant le score des patterns
        """
        try:
            tables = schema_metadata.get('tables', {})
            source_table = tables.get(relation['source'], {})
            target_table = tables.get(relation['target'], {})

            source_col_name = relation['columns']['source'][0] if relation.get('columns', {}).get('source') else None
            target_col_name = relation['columns']['target'][0] if relation.get('columns', {}).get('target') else None

            if not (source_col_name and target_col_name):
                return 0.0

            # Analyse des patterns de nommage
            naming_score = self._calculate_naming_pattern_score(source_col_name, target_col_name)

            # Analyse des statistiques de distribution si disponibles
            source_stats = next((col['statistics'] for col in source_table.get('columns', [])
                                 if col['name'] == source_col_name), {})
            target_stats = next((col['statistics'] for col in target_table.get('columns', [])
                                 if col['name'] == target_col_name), {})

            distribution_score = self._calculate_distribution_similarity(source_stats, target_stats)

            # Moyenne pondérée des scores
            return (naming_score * 0.4 + distribution_score * 0.6)

        except Exception as e:
            logger.error(f"Error calculating data pattern score: {str(e)}")
            return 0.0

    def _calculate_naming_pattern_score(self, source_name: str, target_name: str) -> float:
        """
        Calcule la similarité des noms de colonnes.

        Args:
            source_name: Nom de la colonne source
            target_name: Nom de la colonne cible

        Returns:
            float représentant le score de similarité
        """
        try:
            # Nettoyage des noms
            source_clean = source_name.lower().replace('_', '')
            target_clean = target_name.lower().replace('_', '')

            # Calcul de la distance de Levenshtein
            distance = self._levenshtein_distance(source_clean, target_clean)
            max_length = max(len(source_clean), len(target_clean))

            if max_length == 0:
                return 0.0

            similarity = 1 - (distance / max_length)
            return similarity

        except Exception as e:
            logger.error(f"Error calculating naming pattern score: {str(e)}")
            return 0.0

    def _calculate_distribution_similarity(self, source_stats: Dict, target_stats: Dict) -> float:
        """
        Calcule la similarité des distributions de données.

        Args:
            source_stats: Statistiques de la colonne source
            target_stats: Statistiques de la colonne cible

        Returns:
            float représentant le score de similarité
        """
        try:
            # Si les statistiques ne sont pas disponibles
            if not (source_stats and target_stats):
                return 0.5

            # Comparaison des statistiques de base
            metrics = [
                ('null_percentage', 0.2),
                ('unique_percentage', 0.4),
                ('value_distribution', 0.4)
            ]

            total_score = 0.0
            total_weight = 0.0

            for metric, weight in metrics:
                source_value = source_stats.get(metric)
                target_value = target_stats.get(metric)

                if source_value is not None and target_value is not None:
                    similarity = 1 - abs(source_value - target_value)
                    total_score += similarity * weight
                    total_weight += weight

            if total_weight == 0:
                return 0.5

            return total_score / total_weight

        except Exception as e:
            logger.error(f"Error calculating distribution similarity: {str(e)}")
            return 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calcule la distance de Levenshtein entre deux chaînes.

        Args:
            s1: Première chaîne
            s2: Deuxième chaîne

        Returns:
            int représentant la distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _match_llm_analysis(self, relation: Dict, llm_analysis: Dict) -> Dict:
        """
        Associe l'analyse LLM à une relation spécifique.

        Args:
            relation: Relation à enrichir
            llm_analysis: Analyse LLM complète

        Returns:
            Dict contenant l'analyse LLM correspondante
        """
        try:
            # Création de la clé de correspondance
            relation_key = f"{relation['source']}__{relation['target']}"

            # Recherche de l'analyse correspondante
            if 'relationships' in llm_analysis:
                for analysis in llm_analysis['relationships']:
                    if analysis.get('relation_key') == relation_key:
                        return analysis

            # Analyse par défaut si aucune correspondance trouvée
            return {
                'business_context': 'No specific analysis available',
                'confidence': 0.0,
                'suggestions': []
            }

        except Exception as e:
            logger.error(f"Error matching LLM analysis: {str(e)}")
            return {}