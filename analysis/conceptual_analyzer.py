from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import networkx as nx
from core.config import Config
from core.types import TableInfo, AnalysisResult
from analysis.llm_analyzer import LLMAnalyzer
from storage.result_manager import ResultManager
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConceptualAnalyzer:
    """Analyseur conceptuel de la structure de la base de données."""

    def __init__(self, llm_analyzer: Optional[LLMAnalyzer] = None):
        """
        Initialise l'analyseur conceptuel.

        Args:
            llm_analyzer: Instance de LLMAnalyzer (optionnel)
        """
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()
        self.graph = nx.DiGraph()

    def analyze_schema_concepts(self, schema_metadata: Dict) -> Dict:
        """
        Analyse les concepts métier du schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant l'analyse conceptuelle
        """
        try:
            # Extraction des concepts principaux
            concepts = self._extract_business_concepts(schema_metadata)

            # Identification des relations
            relationships = self._identify_concept_relationships(concepts, schema_metadata)

            # Génération de la hiérarchie
            hierarchy = self._generate_concept_hierarchy(concepts, relationships)

            return {
                'concepts': concepts,
                'relationships': relationships,
                'hierarchy': hierarchy,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
        except Exception as e:
            logger.error(f"Error in schema concept analysis: {str(e)}")
            return {'error': str(e)}

    def _extract_business_concepts(self, schema_metadata: Dict) -> List[Dict]:
        """
        Extrait les concepts métier du schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Liste des concepts identifiés
        """
        prompt = f"""
        Identify and describe the main business concepts in this database schema.
        Schema metadata: {schema_metadata}

        For each concept, provide:
        1. Name and description
        2. Primary business purpose
        3. Key attributes
        4. Business rules
        5. Data quality requirements
        
        Answer in French
        """

        response = self.llm_analyzer.analyze_with_prompt(prompt)
        return self._process_concept_response(response)

    def _identify_concept_relationships(self, concepts: List[Dict], schema_metadata: Dict) -> List[Dict]:
        """
        Identifie les relations entre concepts.

        Args:
            concepts: Liste des concepts
            schema_metadata: Métadonnées du schéma

        Returns:
            Liste des relations identifiées
        """
        relationships = []
        for concept in concepts:
            related_concepts = self._find_related_concepts(concept, concepts, schema_metadata)
            relationships.extend(related_concepts)
        return relationships

    def _generate_concept_hierarchy(self, concepts: List[Dict], relationships: List[Dict]) -> Dict:
        """
        Génère une hiérarchie des concepts.

        Args:
            concepts: Liste des concepts
            relationships: Liste des relations

        Returns:
            Dict représentant la hiérarchie
        """
        G = nx.DiGraph()

        # Ajout des nœuds et arêtes
        for concept in concepts:
            G.add_node(concept['name'], **concept)

        for rel in relationships:
            G.add_edge(rel['source'], rel['target'], **rel.get('attributes', {}))

        # Détection des niveaux hiérarchiques
        hierarchy_levels = self._detect_hierarchy_levels(G)

        return {
            'levels': hierarchy_levels,
            'root_concepts': [node for node, degree in G.in_degree() if degree == 0],
            'leaf_concepts': [node for node, degree in G.out_degree() if degree == 0]
        }

    def _detect_hierarchy_levels(self, G: nx.DiGraph) -> Dict[int, List[str]]:
        """
        Détecte les niveaux hiérarchiques dans le graphe.

        Args:
            G: Graphe des concepts

        Returns:
            Dict des niveaux et leurs concepts
        """
        try:
            levels = {}
            current_level = 0

            # Commence avec les nœuds racines
            nodes = [node for node, degree in G.in_degree() if degree == 0]

            while nodes:
                levels[current_level] = nodes
                next_nodes = []
                for node in nodes:
                    next_nodes.extend(list(G.successors(node)))
                nodes = list(set(next_nodes))  # Éliminer les doublons
                current_level += 1

            return levels

        except Exception as e:
            logger.error(f"Error detecting hierarchy levels: {str(e)}")
            return {}

    def _find_related_concepts(self, concept: Dict, all_concepts: List[Dict],
                               schema_metadata: Dict) -> List[Dict]:
        """
        Trouve les concepts reliés à un concept donné.

        Args:
            concept: Concept source
            all_concepts: Liste de tous les concepts
            schema_metadata: Métadonnées du schéma

        Returns:
            Liste des relations trouvées
        """
        prompt = f"""
        Identify relationships between this concept and other concepts:
        Source concept: {concept}
        Available concepts: {[c['name'] for c in all_concepts]}
        Schema context: {schema_metadata}

        For each relationship identify:
        1. Type of relationship
        2. Cardinality
        3. Business rules
        4. Data dependencies
        5. Quality requirements
        
        Answer in French
        """

        response = self.llm_analyzer.analyze_with_prompt(prompt)
        return self._process_relationship_response(response, concept['name'])