from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
from core.config import Config
from core.types import TableInfo, AnalysisResult
from analysis.llm_analyzer import LLMAnalyzer
from storage.result_manager import ResultManager
import networkx as nx
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analyseur de patterns dans les données."""

    def __init__(self, llm_analyzer: Optional[LLMAnalyzer] = None):
        """
        Initialise l'analyseur de patterns.

        Args:
            llm_analyzer: Instance de LLMAnalyzer (optionnel)
        """
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()

    def analyze_column_dependencies(self, table_metadata: Dict) -> Dict:
        """
        Analyse les dépendances entre colonnes.

        Args:
            table_metadata: Métadonnées de la table

        Returns:
            Dict contenant l'analyse des dépendances
        """
        return self.llm_analyzer.analyze_with_prompt(
            self._create_dependencies_prompt(table_metadata)
        )

    def detect_complex_patterns(self, schema_metadata: Dict) -> Dict:
        """
        Détecte les patterns complexes dans le schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant les patterns détectés
        """
        return self.llm_analyzer.analyze_with_prompt(
            self._create_patterns_prompt(schema_metadata)
        )

    def explain_business_impact(self, patterns: Dict) -> Dict:
        """
        Explique l'impact métier des patterns détectés.

        Args:
            patterns: Patterns détectés

        Returns:
            Dict contenant l'analyse d'impact
        """
        return self.llm_analyzer.analyze_with_prompt(
            self._create_impact_prompt(patterns)
        )

    def _create_dependencies_prompt(self, table_metadata: Dict) -> str:
        """Crée le prompt pour l'analyse des dépendances."""
        return f"""
        Analyze column dependencies in this table.
        Table metadata: {table_metadata}

        Focus on:
        1. Functional dependencies
        2. Correlation patterns
        3. Business rules
        4. Data quality implications
        5. Optimization opportunities
        
        Answer in French
        """

    def _create_patterns_prompt(self, schema_metadata: Dict) -> str:
        """Crée le prompt pour la détection des patterns."""
        return f"""
        Detect complex patterns in this database schema.
        Schema metadata: {schema_metadata}

        Look for:
        1. Data structures patterns
        2. Usage patterns
        3. Access patterns
        4. Update patterns
        5. Integration patterns
        
        Answer in French
        """

    def _create_impact_prompt(self, patterns: Dict) -> str:
        """Crée le prompt pour l'analyse d'impact."""
        return f"""
        Explain the business impact of these detected patterns.
        Patterns: {patterns}

        Consider:
        1. Performance implications
        2. Maintenance impact
        3. Scalability considerations
        4. Business opportunities
        5. Risk factors
        
        Answer in French
        
        """
