from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import os
import json
from core.config import Config
from core.types import TableInfo, AnalysisResult
from analysis.llm_analyzer import LLMAnalyzer
from storage.result_manager import ResultManager
from core.utils import JSONEncoder
import markdown
from jinja2 import Template

logger = logging.getLogger(__name__)

class DocumentationGenerator:
    """Générateur de documentation technique."""

    def __init__(self, llm_analyzer: Optional[LLMAnalyzer] = None):
        """
        Initialise le générateur de documentation.

        Args:
            llm_analyzer: Instance de LLMAnalyzer (optionnel)
        """
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()

    def generate_architecture_doc(self, schema_metadata: Dict) -> Dict:
        """
        Génère la documentation d'architecture.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant la documentation d'architecture
        """
        prompt = f"""
        Generate comprehensive architecture documentation for this database schema.
        Schema metadata: {schema_metadata}

        Include:
        1. Overall architecture
        2. Component relationships
        3. Data flows
        4. Security considerations
        5. Performance characteristics
        
        Answer in French
        """

        return self.llm_analyzer.analyze_with_prompt(prompt)

    def generate_usage_guidelines(self, schema_metadata: Dict) -> Dict:
        """
        Génère le guide d'utilisation.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant les guidelines d'utilisation
        """
        prompt = f"""
        Generate usage guidelines for this database schema.
        Schema metadata: {schema_metadata}

        Cover:
        1. Best practices
        2. Common patterns
        3. Performance tips
        4. Maintenance guidelines
        5. Troubleshooting guides
        
        Answer in French
        """

        return self.llm_analyzer.analyze_with_prompt(prompt)

    def generate_component_explanations(self, schema_metadata: Dict) -> Dict:
        """
        Génère les explications des composants.

        Args:
            schema_metadata: Métadonnées du schéma

        Returns:
            Dict contenant les explications des composants
        """
        prompt = f"""
        Generate detailed explanations for each database component.
        Schema metadata: {schema_metadata}

        For each component:
        1. Purpose and function
        2. Technical details
        3. Dependencies
        4. Usage examples
        5. Maintenance notes
        
        Answer in French
        """

        return self.llm_analyzer.analyze_with_prompt(prompt)