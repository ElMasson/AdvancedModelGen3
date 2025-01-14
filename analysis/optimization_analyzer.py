from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import numpy as np
from core.config import Config
from core.types import TableInfo, AnalysisResult
from analysis.llm_analyzer import LLMAnalyzer
from storage.result_manager import ResultManager
from sqlalchemy import inspect
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptimizationAnalyzer:
    """Analyseur d'optimisation de la base de données."""

    def __init__(self, llm_analyzer: Optional[LLMAnalyzer] = None):
        """
        Initialise l'analyseur d'optimisation.

        Args:
            llm_analyzer: Instance de LLMAnalyzer (optionnel)
        """
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()

    def analyze_schema_efficiency(self, schema_metadata: Dict) -> Dict:
        """
        Analyse l'efficacité globale du schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant l'analyse d'efficacité
        """
        try:
            # Analyse de base
            base_analysis = self._analyze_base_metrics(schema_metadata)

            # Analyse LLM
            llm_analysis = self.llm_analyzer.analyze_with_prompt(
                self._create_efficiency_prompt(schema_metadata)
            )

            # Calcul du score global
            efficiency_score = self._calculate_efficiency_score(base_analysis, llm_analysis)

            return {
                'overall_score': efficiency_score,
                'base_analysis': base_analysis,
                'llm_analysis': llm_analysis,
                'metrics': {
                    'normalization_score': self._calculate_normalization_score(schema_metadata),
                    'index_efficiency': self._calculate_index_efficiency(schema_metadata),
                    'storage_efficiency': self._calculate_storage_efficiency(schema_metadata),
                    'query_efficiency': self._calculate_query_efficiency(schema_metadata)
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
        except Exception as e:
            logger.error(f"Error in schema efficiency analysis: {str(e)}")
            return {'error': str(e)}

    def suggest_optimizations(self, analysis_results: Dict) -> Dict:
        """
        Suggère des optimisations basées sur l'analyse.

        Args:
            analysis_results: Résultats de l'analyse

        Returns:
            Dict contenant les suggestions d'optimisation
        """
        try:
            # Génération des suggestions via LLM
            suggestions = self.llm_analyzer.analyze_with_prompt(
                self._create_optimization_prompt(analysis_results)
            )

            # Validation et enrichissement des suggestions
            validated_suggestions = self._validate_optimization_suggestions(suggestions)

            return {
                'suggestions': validated_suggestions,
                'priorities': self._prioritize_suggestions(validated_suggestions),
                'implementation_plan': self._create_implementation_plan(validated_suggestions),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'based_on_analysis': analysis_results.get('metadata', {}).get('timestamp')
                }
            }
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {str(e)}")
            return {'error': str(e)}

    def explain_optimization_impacts(self, suggestions: Dict) -> Dict:
        """
        Explique l'impact des optimisations suggérées.

        Args:
            suggestions: Suggestions d'optimisation

        Returns:
            Dict contenant l'analyse d'impact
        """
        try:
            # Analyse d'impact via LLM
            impacts = self.llm_analyzer.analyze_with_prompt(
                self._create_impact_prompt(suggestions)
            )

            # Enrichissement avec métriques quantitatives
            enriched_impacts = self._enrich_impact_analysis(impacts)

            return {
                'impacts': enriched_impacts,
                'summary': self._create_impact_summary(enriched_impacts),
                'risk_assessment': self._assess_implementation_risks(enriched_impacts),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'based_on_suggestions': suggestions.get('metadata', {}).get('timestamp')
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing optimization impacts: {str(e)}")
            return {'error': str(e)}

    def _analyze_base_metrics(self, schema_metadata: Dict) -> Dict:
        """
        Analyse les métriques de base du schéma.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant les métriques de base
        """
        return {
            'table_metrics': self._analyze_table_metrics(schema_metadata),
            'index_metrics': self._analyze_index_metrics(schema_metadata),
            'column_metrics': self._analyze_column_metrics(schema_metadata),
            'relationship_metrics': self._analyze_relationship_metrics(schema_metadata)
        }

    def _calculate_efficiency_score(self, base_analysis: Dict, llm_analysis: Dict) -> float:
        """
        Calcule le score d'efficacité global.

        Args:
            base_analysis: Analyse de base
            llm_analysis: Analyse LLM

        Returns:
            float représentant le score d'efficacité
        """
        scores = {
            'table_efficiency': self._calculate_table_efficiency(base_analysis),
            'index_efficiency': self._calculate_index_efficiency_score(base_analysis),
            'column_efficiency': self._calculate_column_efficiency(base_analysis),
            'relationship_efficiency': self._calculate_relationship_efficiency(base_analysis),
            'llm_score': self._extract_llm_efficiency_score(llm_analysis)
        }

        weights = {
            'table_efficiency': 0.2,
            'index_efficiency': 0.2,
            'column_efficiency': 0.2,
            'relationship_efficiency': 0.2,
            'llm_score': 0.2
        }

        return sum(score * weights[metric] for metric, score in scores.items())

    def _validate_optimization_suggestions(self, suggestions: Dict) -> Dict:
        """
        Valide et enrichit les suggestions d'optimisation.

        Args:
            suggestions: Suggestions à valider

        Returns:
            Dict contenant les suggestions validées
        """
        validated = {}
        for category, category_suggestions in suggestions.items():
            validated[category] = []
            for suggestion in category_suggestions:
                if self._is_suggestion_valid(suggestion):
                    enriched_suggestion = {
                        **suggestion,
                        'feasibility': self._calculate_feasibility(suggestion),
                        'complexity': self._estimate_complexity(suggestion),
                        'impact': self._estimate_impact(suggestion),
                        'prerequisites': self._identify_prerequisites(suggestion),
                        'risks': self._identify_risks(suggestion)
                    }
                    validated[category].append(enriched_suggestion)
        return validated

    def _create_implementation_plan(self, suggestions: Dict) -> Dict:
        """
        Crée un plan d'implémentation pour les suggestions.

        Args:
            suggestions: Suggestions validées

        Returns:
            Dict contenant le plan d'implémentation
        """
        try:
            # Organisation par phases
            phases = self._organize_implementation_phases(suggestions)

            # Estimation des ressources
            resources = self._estimate_required_resources(phases)

            # Création du planning
            schedule = self._create_implementation_schedule(phases, resources)

            return {
                'phases': phases,
                'resources': resources,
                'schedule': schedule,
                'dependencies': self._identify_phase_dependencies(phases),
                'risks': self._identify_implementation_risks(phases),
                'monitoring': self._create_monitoring_plan(phases)
            }
        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}")
            return {'error': str(e)}

    def _assess_implementation_risks(self, impacts: Dict) -> Dict:
        """
        Évalue les risques liés à l'implémentation.

        Args:
            impacts: Analyse d'impact

        Returns:
            Dict contenant l'évaluation des risques
        """
        risk_categories = {
            'technical': self._assess_technical_risks(impacts),
            'business': self._assess_business_risks(impacts),
            'operational': self._assess_operational_risks(impacts),
            'security': self._assess_security_risks(impacts),
            'performance': self._assess_performance_risks(impacts)
        }

        return {
            'categories': risk_categories,
            'overall_risk_score': self._calculate_overall_risk_score(risk_categories),
            'mitigation_strategies': self._generate_risk_mitigation_strategies(risk_categories),
            'monitoring_recommendations': self._generate_risk_monitoring_recommendations(risk_categories)
        }

    def _create_monitoring_plan(self, phases: Dict) -> Dict:
        """
        Crée un plan de monitoring pour l'implémentation.

        Args:
            phases: Phases d'implémentation

        Returns:
            Dict contenant le plan de monitoring
        """
        return {
            'metrics': self._define_monitoring_metrics(phases),
            'alerts': self._define_monitoring_alerts(phases),
            'reporting': self._define_reporting_structure(phases),
            'thresholds': self._define_monitoring_thresholds(phases),
            'procedures': self._define_monitoring_procedures(phases)
        }

        # analysis/optimization_analyzer.py (suite)

        def _prioritize_suggestions(self, suggestions: Dict) -> List[Dict]:
            """
            Priorise les suggestions d'optimisation.

            Args:
                suggestions: Suggestions validées

            Returns:
                Liste des suggestions priorisées
            """
            prioritized = []
            for category, category_suggestions in suggestions.items():
                for suggestion in category_suggestions:
                    priority_score = self._calculate_priority_score(suggestion)
                    prioritized.append({
                        **suggestion,
                        'category': category,
                        'priority_score': priority_score,
                        'priority_level': self._determine_priority_level(priority_score),
                        'implementation_order': self._determine_implementation_order(
                            suggestion, priority_score
                        )
                    })

            # Tri par score de priorité
            return sorted(prioritized, key=lambda x: x['priority_score'], reverse=True)

        def _calculate_priority_score(self, suggestion: Dict) -> float:
            """
            Calcule le score de priorité d'une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score de priorité
            """
            weights = {
                'impact': 0.4,
                'effort': 0.2,
                'risk': 0.2,
                'urgency': 0.2
            }

            scores = {
                'impact': self._evaluate_impact_score(suggestion),
                'effort': self._evaluate_effort_score(suggestion),
                'risk': self._evaluate_risk_score(suggestion),
                'urgency': self._evaluate_urgency_score(suggestion)
            }

            return sum(score * weights[factor] for factor, score in scores.items())

        def _determine_priority_level(self, priority_score: float) -> str:
            """
            Détermine le niveau de priorité basé sur le score.

            Args:
                priority_score: Score de priorité

            Returns:
                str représentant le niveau de priorité
            """
            if priority_score >= 0.8:
                return "CRITICAL"
            elif priority_score >= 0.6:
                return "HIGH"
            elif priority_score >= 0.4:
                return "MEDIUM"
            else:
                return "LOW"

        def _determine_implementation_order(self, suggestion: Dict, priority_score: float) -> int:
            """
            Détermine l'ordre d'implémentation optimal.

            Args:
                suggestion: Suggestion à évaluer
                priority_score: Score de priorité

            Returns:
                int représentant l'ordre d'implémentation
            """
            base_order = int(priority_score * 100)

            # Ajustement basé sur les dépendances
            dependencies = suggestion.get('prerequisites', [])
            if dependencies:
                base_order += len(dependencies) * 10

            # Ajustement basé sur la complexité
            complexity = suggestion.get('complexity', 'MEDIUM')
            complexity_adjustments = {
                'LOW': 0,
                'MEDIUM': 5,
                'HIGH': 10
            }
            base_order += complexity_adjustments.get(complexity, 0)

            return base_order

        def _evaluate_impact_score(self, suggestion: Dict) -> float:
            """
            Évalue l'impact d'une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score d'impact
            """
            impact_factors = {
                'performance_improvement': self._evaluate_performance_impact(suggestion),
                'maintenance_improvement': self._evaluate_maintenance_impact(suggestion),
                'scalability_improvement': self._evaluate_scalability_impact(suggestion),
                'cost_reduction': self._evaluate_cost_impact(suggestion)
            }

            weights = {
                'performance_improvement': 0.4,
                'maintenance_improvement': 0.2,
                'scalability_improvement': 0.2,
                'cost_reduction': 0.2
            }

            return sum(score * weights[factor] for factor, score in impact_factors.items())

        def _evaluate_effort_score(self, suggestion: Dict) -> float:
            """
            Évalue l'effort requis pour une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score d'effort (inversé)
            """
            effort_factors = {
                'implementation_time': self._estimate_implementation_time(suggestion),
                'resource_requirements': self._estimate_resource_requirements(suggestion),
                'technical_complexity': self._estimate_technical_complexity(suggestion),
                'testing_requirements': self._estimate_testing_requirements(suggestion)
            }

            # Score inversé car un effort plus élevé devrait réduire la priorité
            raw_score = sum(effort_factors.values()) / len(effort_factors)
            return 1 - (raw_score / 10)  # Normalisation sur une échelle de 0 à 1

        def _evaluate_risk_score(self, suggestion: Dict) -> float:
            """
            Évalue le risque associé à une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score de risque (inversé)
            """
            risk_factors = {
                'implementation_risk': self._assess_implementation_risk(suggestion),
                'business_risk': self._assess_business_risk(suggestion),
                'security_risk': self._assess_security_risk(suggestion),
                'performance_risk': self._assess_performance_risk(suggestion)
            }

            # Score inversé car un risque plus élevé devrait réduire la priorité
            raw_score = sum(risk_factors.values()) / len(risk_factors)
            return 1 - (raw_score / 10)

        def _evaluate_urgency_score(self, suggestion: Dict) -> float:
            """
            Évalue l'urgence d'une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score d'urgence
            """
            urgency_factors = {
                'current_performance_impact': self._evaluate_current_performance_impact(suggestion),
                'business_criticality': self._evaluate_business_criticality(suggestion),
                'technical_debt': self._evaluate_technical_debt(suggestion),
                'growth_impact': self._evaluate_growth_impact(suggestion)
            }

            weights = {
                'current_performance_impact': 0.3,
                'business_criticality': 0.3,
                'technical_debt': 0.2,
                'growth_impact': 0.2
            }

            return sum(score * weights[factor] for factor, score in urgency_factors.items())

        def _evaluate_performance_impact(self, suggestion: Dict) -> float:
            """
            Évalue l'impact sur les performances.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant l'impact sur les performances
            """
            impact_metrics = {
                'query_time_reduction': self._estimate_query_time_reduction(suggestion),
                'resource_usage_reduction': self._estimate_resource_usage_reduction(suggestion),
                'throughput_improvement': self._estimate_throughput_improvement(suggestion),
                'latency_reduction': self._estimate_latency_reduction(suggestion)
            }

            return sum(impact_metrics.values()) / len(impact_metrics)

        def _assess_implementation_risk(self, suggestion: Dict) -> float:
            """
            Évalue le risque d'implémentation.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le niveau de risque
            """
            risk_factors = {
                'complexity_risk': self._evaluate_complexity_risk(suggestion),
                'dependency_risk': self._evaluate_dependency_risk(suggestion),
                'rollback_risk': self._evaluate_rollback_risk(suggestion),
                'testing_risk': self._evaluate_testing_risk(suggestion)
            }

            return sum(risk_factors.values()) / len(risk_factors)

        def _evaluate_technical_debt(self, suggestion: Dict) -> float:
            """
            Évalue l'impact sur la dette technique.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant l'impact sur la dette technique
            """
            debt_factors = {
                'code_quality_impact': self._evaluate_code_quality_impact(suggestion),
                'maintainability_impact': self._evaluate_maintainability_impact(suggestion),
                'scalability_impact': self._evaluate_scalability_impact_on_debt(suggestion),
                'documentation_impact': self._evaluate_documentation_impact(suggestion)
            }

            weights = {
                'code_quality_impact': 0.3,
                'maintainability_impact': 0.3,
                'scalability_impact': 0.2,
                'documentation_impact': 0.2
            }

            return sum(score * weights[factor] for factor, score in debt_factors.items())

        def _calculate_complexity_score(self, suggestion: Dict) -> float:
            """
            Calcule un score de complexité pour une suggestion.

            Args:
                suggestion: Suggestion à évaluer

            Returns:
                float représentant le score de complexité
            """
            complexity_factors = {
                'implementation_complexity': self._evaluate_implementation_complexity(suggestion),
                'testing_complexity': self._evaluate_testing_complexity(suggestion),
                'integration_complexity': self._evaluate_integration_complexity(suggestion),
                'maintenance_complexity': self._evaluate_maintenance_complexity(suggestion)
            }

            weights = {
                'implementation_complexity': 0.4,
                'testing_complexity': 0.2,
                'integration_complexity': 0.2,
                'maintenance_complexity': 0.2
            }

            return sum(score * weights[factor] for factor, score in complexity_factors.items())

        def _create_efficiency_prompt(self, schema_metadata: Dict) -> str:
            """
            Crée le prompt pour l'analyse d'efficacité.

            Args:
                schema_metadata: Métadonnées du schéma

            Returns:
                str contenant le prompt
            """
            return f"""
            Analyze the efficiency of this database schema and provide detailed insights.
            Schema metadata: {schema_metadata}

            Focus on:
            1. Schema design efficiency
            2. Query optimization opportunities
            3. Storage optimization potential
            4. Performance bottlenecks
            5. Maintenance challenges

            Consider:
            - Current performance metrics
            - Scalability factors
            - Resource utilization
            - Maintenance overhead
            - Technical debt indicators

            Provide concrete recommendations for:
            1. Immediate optimizations
            2. Long-term improvements
            3. Risk mitigation strategies
            4. Implementation priorities
            5. Expected benefits
            
            Answer in French
            """

        def _create_optimization_prompt(self, analysis_results: Dict) -> str:
            """
            Crée le prompt pour les suggestions d'optimisation.

            Args:
                analysis_results: Résultats de l'analyse

            Returns:
                str contenant le prompt
            """
            return f"""
            Based on the analysis results, suggest optimizations for this database schema.
            Analysis results: {analysis_results}

            Provide recommendations for:
            1. Schema optimizations
            2. Query performance improvements
            3. Storage efficiency enhancements
            4. Maintenance simplification
            5. Technical debt reduction

            For each suggestion, include:
            - Detailed description
            - Implementation complexity
            - Expected benefits
            - Potential risks
            - Prerequisites
            - Resource requirements

            Prioritize suggestions based on:
            - Impact on performance
            - Implementation effort
            - Risk level
            - Business value
            - Urgency
            
            Answer in French
            """

        def _create_impact_prompt(self, suggestions: Dict) -> str:
            """
            Crée le prompt pour l'analyse d'impact.

            Args:
                suggestions: Suggestions d'optimisation

            Returns:
                str contenant le prompt
            """
            return f"""
            Analyze the potential impacts of implementing these optimization suggestions.
            Suggestions: {suggestions}

            For each suggestion, evaluate:
            1. Performance impact
            2. Resource utilization impact
            3. Maintenance impact
            4. Business process impact
            5. User experience impact

            Consider:
            - Short-term vs long-term effects
            - Direct and indirect consequences
            - Risk factors
            - Dependencies
            - Resource requirements

            Provide:
            1. Detailed impact analysis
            2. Risk assessment
            3. Mitigation strategies
            4. Implementation recommendations
            5. Monitoring suggestions
            
            Answer in French
            """