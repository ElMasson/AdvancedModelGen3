from typing import Dict, List, Optional, Any, Union, Set, List
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from core.config import Config
from core.types import TableInfo, AnalysisResult
import statistics
from collections import defaultdict
import json
from scipy import stats
import re


logger = logging.getLogger(__name__)

def calculate_efficiency_metrics(table_metadata: Dict) -> Dict:
    """
    Calcule les métriques d'efficacité pour une table.

    Args:
        table_metadata: Métadonnées de la table

    Returns:
        Dict contenant les métriques d'efficacité
    """
    return {
        'storage_efficiency': calculate_storage_efficiency(table_metadata),
        'index_efficiency': calculate_index_efficiency(table_metadata),
        'query_efficiency': calculate_query_efficiency(table_metadata),
        'normalization_score': calculate_normalization_score(table_metadata),
        'integrity_score': calculate_integrity_score(table_metadata)
    }


def calculate_storage_efficiency(table_metadata: Dict) -> float:
    """
    Calcule l'efficacité du stockage.

    Args:
        table_metadata: Métadonnées de la table

    Returns:
        float représentant le score d'efficacité
    """
    try:
        # Calcul basé sur plusieurs facteurs
        factors = {
            'data_type_optimization': evaluate_data_type_optimization(table_metadata),
            'compression_ratio': evaluate_compression_ratio(table_metadata),
            'space_utilization': evaluate_space_utilization(table_metadata),
            'fragmentation': evaluate_fragmentation(table_metadata)
        }

        weights = {
            'data_type_optimization': 0.3,
            'compression_ratio': 0.3,
            'space_utilization': 0.2,
            'fragmentation': 0.2
        }

        return sum(score * weights[factor] for factor, score in factors.items())

    except Exception as e:
        logger.error(f"Error calculating storage efficiency: {str(e)}")
        return 0.0


def calculate_query_efficiency(table_metadata: Dict) -> float:
    """
    Calcule l'efficacité des requêtes.

    Args:
        table_metadata: Métadonnées de la table

    Returns:
        float représentant le score d'efficacité
    """
    try:
        # Analyse des facteurs d'efficacité des requêtes
        factors = {
            'index_usage': evaluate_index_usage(table_metadata),
            'query_patterns': evaluate_query_patterns(table_metadata),
            'join_efficiency': evaluate_join_efficiency(table_metadata),
            'filter_efficiency': evaluate_filter_efficiency(table_metadata)
        }

        weights = {
            'index_usage': 0.3,
            'query_patterns': 0.3,
            'join_efficiency': 0.2,
            'filter_efficiency': 0.2
        }

        return sum(score * weights[factor] for factor, score in factors.items())

    except Exception as e:
        logger.error(f"Error calculating query efficiency: {str(e)}")
        return 0.0


class BusinessRuleValidator:
    """Validateur de règles métier."""

    # Mots-clés indiquant des opérations simples à ignorer
    BASIC_OPERATIONS: Set[str] = {
        'count', 'sum', 'average', 'avg', 'group by', 'order by',
        'min', 'max', 'distinct', 'like', 'in'
    }

    # Mots-clés indiquant des calculs ou filtres complexes
    COMPLEX_INDICATORS: Set[str] = {
        'ratio', 'percentage', 'rate', 'threshold', 'criteria',
        'condition', 'efficiency', 'performance', 'margin',
        'conversion', 'retention', 'growth', 'trend'
    }

    # Filtres business significatifs
    SIGNIFICANT_FILTERS: Set[str] = {
        'status', 'phase', 'category', 'priority',
        'risk', 'compliance', 'validation', 'approval',
        'eligibility', 'qualification'
    }

    @classmethod
    def is_significant_rule(cls, rule: Dict) -> bool:
        """
        Vérifie si une règle représente une logique métier significative.

        Args:
            rule: Dictionnaire contenant la règle à valider

        Returns:
            bool: True si la règle est significative
        """
        try:
            description = rule.get('description', '').lower()
            context = rule.get('context', '').lower()
            justification = rule.get('business_justification', '').lower()

            # Vérifier la présence d'une justification business
            if not justification:
                return False

            # 1. Détecter les calculs complexes
            if any(indicator in description for indicator in cls.COMPLEX_INDICATORS):
                return True

            # 2. Vérifier les filtres significatifs
            if any(filter_word in description for filter_word in cls.SIGNIFICANT_FILTERS):
                return True

            # 3. Détecter les opérations mathématiques complexes
            if cls._has_complex_calculation(description):
                return True

            # 4. Vérifier la présence d'opérations basiques sans contexte business
            if cls._is_basic_operation(description) and not cls._has_business_context(context):
                return False

            # 5. Vérifier les conditions temporelles significatives
            if cls._has_significant_temporal_logic(description):
                return True

            # 6. Vérifier les règles de validation business
            if cls._has_business_validation_rules(description, context):
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating business rule: {str(e)}")
            return False

    @classmethod
    def _has_complex_calculation(cls, text: str) -> bool:
        """Détecte la présence de calculs complexes."""
        # Patterns pour les calculs complexes
        calc_patterns = [
            r'ratio|percentage|rate',
            r'average\s+over|running\s+total',
            r'year[\-_]over[\-_]year|month[\-_]over[\-_]month',
            r'growth\s+rate|conversion\s+rate',
            r'weighted|normalized|adjusted'
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in calc_patterns)

    @classmethod
    def _is_basic_operation(cls, text: str) -> bool:
        """Vérifie si la règle ne contient que des opérations basiques."""
        return any(op in text for op in cls.BASIC_OPERATIONS) and \
            not any(indicator in text for indicator in cls.COMPLEX_INDICATORS)

    @classmethod
    def _has_business_context(cls, text: str) -> bool:
        """Vérifie la présence d'un contexte business significatif."""
        business_terms = {
            'business', 'strategy', 'policy', 'requirement',
            'compliance', 'regulation', 'standard', 'rule'
        }
        return any(term in text for term in business_terms)

    @classmethod
    def _has_significant_temporal_logic(cls, text: str) -> bool:
        """Détecte la présence de logique temporelle significative."""
        temporal_patterns = [
            r'trend|evolution|progression',
            r'historical|forecast|prediction',
            r'period\s+comparison|time\s+analysis',
            r'seasonal|cyclical|periodic'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in temporal_patterns)

    @classmethod
    def _has_business_validation_rules(cls, description: str, context: str) -> bool:
        """Vérifie la présence de règles de validation business."""
        validation_patterns = [
            r'must|should|required|mandatory',
            r'validation|verification|check',
            r'threshold|limit|boundary',
            r'compliance|conform|standard'
        ]
        text = f"{description} {context}"
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in validation_patterns)

    @classmethod
    def filter_rules(cls, rules: List[Dict]) -> List[Dict]:
        """
        Filtre une liste de règles pour ne garder que les significatives.

        Args:
            rules: Liste des règles à filtrer

        Returns:
            Liste des règles significatives
        """
        return [rule for rule in rules if cls.is_significant_rule(rule)]

    @classmethod
    def filter_joins(cls, joins: List[Dict]) -> List[Dict]:
        """
        Filtre les jointures pour ne garder que celles avec une signification business.

        Args:
            joins: Liste des jointures à filtrer

        Returns:
            Liste des jointures significatives
        """
        return [
            join for join in joins
            if join.get('justification') and
               not any(word in join['justification'].lower()
                       for word in ['simple', 'basic', 'standard'])
        ]