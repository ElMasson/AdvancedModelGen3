# analysis/sql_rules_analyzer.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import openai
import json
import logging
from datetime import datetime
from core.config import Config
import time
from .utils import BusinessRuleValidator

logger = logging.getLogger(__name__)

@dataclass
class BusinessRule:
    """Structure pour une règle métier."""
    description: str
    impact: str
    tables: List[str]
    source_queries: List[str]
    rule_type: str  # 'table' ou 'column'
    confidence: float = 1.0
    related_columns: Optional[List[str]] = None
    context: Optional[str] = None

@dataclass
class JoinInfo:
    """Structure pour une relation de jointure."""
    source_table: str
    target_table: str
    business_meaning: str
    importance: str
    source_queries: List[str]
    join_type: Optional[str] = None
    source_column: Optional[str] = None
    target_column: Optional[str] = None

@dataclass
class AnalysisResult:
    """Structure pour les résultats d'analyse."""
    table_rules: List[BusinessRule]
    column_rules: Dict[str, List[BusinessRule]]
    joins: List[JoinInfo]
    timestamp: datetime
    error: Optional[str] = None


class SQLRulesAnalyzer:
    """Analyseur de règles métier depuis les exemples SQL."""

    def __init__(self):
        """Initialise l'analyseur avec le client OpenAI."""
        try:
            if not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in configuration")

            self.openai_client = openai.OpenAI(
                api_key=Config.OPENAI_API_KEY,
                timeout=Config.LLM_TIMEOUT
            )

            logger.info("SQLRulesAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SQLRulesAnalyzer: {str(e)}")
            raise

    def _structure_analysis(self, raw_analysis: Dict, source_query: str) -> AnalysisResult:
        """Structure l'analyse brute en format standardisé."""
        try:
            # Conversion des règles tables
            table_rules = [
                BusinessRule(
                    description=rule['description'],
                    impact=rule['impact'],
                    tables=rule['tables'],
                    source_queries=[source_query] + rule.get('source_queries', []),
                    rule_type='table',
                    confidence=1.0
                )
                for rule in raw_analysis.get('table_rules', [])
            ]

            # Conversion des règles colonnes
            column_rules = {
                col_name: [
                    BusinessRule(
                        description=rule['description'],
                        impact=rule.get('impact', 'Unknown'),
                        tables=[],
                        source_queries=[source_query] + rule.get('source_queries', []),
                        rule_type='column',
                        confidence=1.0,
                        context=rule.get('context'),
                        related_columns=rule.get('related_columns', [])
                    )
                    for rule in rules
                ]
                for col_name, rules in raw_analysis.get('column_rules', {}).items()
            }

            # Conversion des jointures
            joins = [
                JoinInfo(
                    source_table=join['source_table'],
                    target_table=join['target_table'],
                    business_meaning=join['business_meaning'],
                    importance=join['importance'],
                    source_queries=[source_query] + join.get('source_queries', []),
                    join_type=join.get('join_type'),
                    source_column=join.get('source_column'),
                    target_column=join.get('target_column')
                )
                for join in raw_analysis.get('joins', [])
            ]

            return AnalysisResult(
                table_rules=table_rules,
                column_rules=column_rules,
                joins=joins,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error structuring analysis: {str(e)}")
            return AnalysisResult(
                table_rules=[],
                column_rules={},
                joins=[],
                timestamp=datetime.now(),
                error=str(e)
            )

    def analyze_query(self, query: str, question: str) -> Dict[str, Any]:
        """
        Analyse une requête SQL pour extraire les règles métier significatives.

        Args:
            query: Requête SQL à analyser
            question: Question métier associée

        Returns:
            Dict contenant les règles métier significatives
        """
        try:
            prompt = """Analysez cette requête SQL et sa question métier pour extraire uniquement les règles de gestion significatives.

FOCUS EXCLUSIVEMENT sur:
1. Les calculs métier complexes (ex: ratios spécifiques, métriques personnalisées, formules métier)
2. Les filtres significatifs (ex: conditions de statut, seuils métier, périodes avec signification business)
3. Les relations de données représentant des concepts métier importants

IGNORER et NE PAS INCLURE:
- Les agrégations simples (COUNT, SUM, AVG)
- Les groupements basiques
- Les jointures standard pour la récupération de données
- Les filtres de date sans contexte métier
- Les clauses WHERE basiques sans signification métier

Question Métier: {question}
Requête SQL: {query}

Fournir uniquement les règles de gestion significatives au format JSON suivant:

{
    "table_rules": [
        {
            "description": "Description de la règle métier complexe ou condition significative",
            "impact": "Explication de l'importance et de l'impact métier",
            "tables": ["tables_concernées"],
            "rule_type": "calcul|filtre|seuil",
            "justification_metier": "Expliquer pourquoi c'est une règle métier significative"
        }
    ],
    "column_rules": {
        "nom_colonne": [
            {
                "description": "Description de la règle concernant les calculs ou conditions significatives",
                "contexte": "Contexte métier clair",
                "colonnes_liees": ["colonnes"],
                "rule_type": "calcul|filtre|seuil",
                "justification_metier": "Expliquer pourquoi c'est une règle métier significative"
            }
        ]
    },
    "joins": [
        {
            "table_source": "table1",
            "table_cible": "table2",
            "signification_metier": "Description claire de la signification métier de la relation",
            "importance": "Expliquer pourquoi cette relation est critique pour le métier",
            "justification": "Pourquoi ce n'est pas une simple jointure technique"
        }
    ]
}

IMPORTANT:
- N'inclure que les règles représentant une véritable logique métier
- Exclure les détails techniques d'implémentation
- Chaque règle doit avoir une signification métier claire
- Si aucune règle significative n'est trouvée, retourner des tableaux vides"""

            max_retries = Config.LLM_MAX_RETRIES
            base_timeout = Config.LLM_TIMEOUT

            for attempt in range(max_retries):
                try:
                    current_timeout = base_timeout * (attempt + 1)

                    # Appel OpenAI avec timeout ajusté
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system",
                             "content": "You are a business analyst expert in understanding SQL queries and extracting business rules. Always respond with valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        timeout=current_timeout
                    )

                    # Nettoyage et extraction du JSON
                    response_text = response.choices[0].message.content.strip()

                    # Suppression des backticks s'ils sont présents
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.startswith('```'):
                        response_text = response_text[3:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]

                    response_text = response_text.strip()

                    try:
                        # Parse du JSON
                        analysis = json.loads(response_text)

                        # Validation de la structure
                        required_keys = ['table_rules', 'column_rules', 'joins']
                        if not all(key in analysis for key in required_keys):
                            missing_keys = [key for key in required_keys if key not in analysis]
                            raise ValueError(f"Missing required keys in response: {missing_keys}")

                        # Filtrage des règles non significatives
                        analysis['table_rules'] = BusinessRuleValidator.filter_rules(
                            analysis.get('table_rules', [])
                        )

                        # Filtrage des règles de colonnes
                        filtered_column_rules = {}
                        for col, rules in analysis.get('column_rules', {}).items():
                            significant_rules = BusinessRuleValidator.filter_rules(rules)
                            if significant_rules:
                                filtered_column_rules[col] = significant_rules
                        analysis['column_rules'] = filtered_column_rules

                        # Filtrage des jointures
                        analysis['joins'] = BusinessRuleValidator.filter_joins(
                            analysis.get('joins', [])
                        )

                        # Ajout des métadonnées
                        analysis['metadata'] = {
                            'timestamp': datetime.now().isoformat(),
                            'source_query': query,
                            'source_question': question,
                            'significant_rules_count': (
                                    len(analysis['table_rules']) +
                                    sum(len(rules) for rules in analysis['column_rules'].values())
                            )
                        }

                        # Ajout des source_queries à toutes les règles
                        for rule in analysis['table_rules']:
                            rule['source_queries'] = [query]

                        for rules in analysis['column_rules'].values():
                            for rule in rules:
                                rule['source_queries'] = [query]

                        for join in analysis['joins']:
                            join['source_queries'] = [query]

                        logger.info(
                            f"Analysis completed: {analysis['metadata']['significant_rules_count']} significant rules found")
                        return analysis

                    except json.JSONDecodeError as json_err:
                        logger.warning(
                            f"JSON parsing error on attempt {attempt + 1}: {str(json_err)}\nResponse: {response_text[:200]}...")
                        if attempt == max_retries - 1:
                            raise ValueError(f"Failed to parse JSON response after {max_retries} attempts")
                        time.sleep(min(2 ** attempt, 10))
                        continue

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    continue

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                'table_rules': [],
                'column_rules': {},
                'joins': [],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'source_query': query,
                    'source_question': question,
                    'significant_rules_count': 0
                }
            }

    def _structure_analysis(self, raw_analysis: Dict, source_query: str) -> Dict[str, Any]:
        """Structure l'analyse brute en format standardisé."""
        try:
            # Ajout de la requête source à toutes les règles
            for rule in raw_analysis.get('table_rules', []):
                if 'source_queries' not in rule:
                    rule['source_queries'] = [source_query]

            for rules in raw_analysis.get('column_rules', {}).values():
                for rule in rules:
                    if 'source_queries' not in rule:
                        rule['source_queries'] = [source_query]

            for join in raw_analysis.get('joins', []):
                if 'source_queries' not in join:
                    join['source_queries'] = [source_query]

            return raw_analysis

        except Exception as e:
            logger.error(f"Error structuring analysis: {str(e)}")
            return {
                'table_rules': [],
                'column_rules': {},
                'joins': [],
                'error': str(e)
            }



    def merge_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fusionne plusieurs analyses en une seule vue consolidée."""
        try:
            merged = {
                'table_rules': [],
                'column_rules': {},
                'joins': [],
                'error': None
            }

            # Fusion des règles de table
            seen_descriptions = set()
            for analysis in analyses:
                for rule in analysis.get('table_rules', []):
                    if rule['description'] not in seen_descriptions:
                        merged['table_rules'].append(rule)
                        seen_descriptions.add(rule['description'])
                    else:
                        # Mise à jour des sources pour règles existantes
                        existing_rule = next(r for r in merged['table_rules']
                                             if r['description'] == rule['description'])
                        existing_rule['source_queries'].extend(
                            rule.get('source_queries', [])
                        )

            # Fusion des règles de colonnes
            for analysis in analyses:
                for col, rules in analysis.get('column_rules', {}).items():
                    if col not in merged['column_rules']:
                        merged['column_rules'][col] = []

                    for rule in rules:
                        existing = next(
                            (r for r in merged['column_rules'][col]
                             if r['description'] == rule['description']),
                            None
                        )
                        if existing:
                            existing['source_queries'].extend(
                                rule.get('source_queries', [])
                            )
                        else:
                            merged['column_rules'][col].append(rule)

            # Fusion des jointures
            seen_joins = set()
            for analysis in analyses:
                for join in analysis.get('joins', []):
                    join_key = (join['source_table'], join['target_table'])
                    if join_key not in seen_joins:
                        merged['joins'].append(join)
                        seen_joins.add(join_key)
                    else:
                        # Mise à jour des sources pour jointures existantes
                        existing_join = next(j for j in merged['joins']
                                             if (j['source_table'], j['target_table']) == join_key)
                        existing_join['source_queries'].extend(
                            join.get('source_queries', [])
                        )

            return merged

        except Exception as e:
            logger.error(f"Error merging analyses: {str(e)}")
            return {
                'table_rules': [],
                'column_rules': {},
                'joins': [],
                'error': str(e)
            }