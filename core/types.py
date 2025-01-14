# core/types.py
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

@dataclass
class ColumnInfo:
    name: str
    type: str
    description: Optional[str] = None
    statistics: Optional[Dict] = None

@dataclass
class TableInfo:
    name: str
    schema: str
    columns: List[ColumnInfo]
    row_count: int
    metadata: Optional[Dict] = None

@dataclass
class AnalysisResult:
    table_info: TableInfo
    llm_analysis: Dict
    relationships: List[Dict]
    performance_metrics: Optional[Dict] = None