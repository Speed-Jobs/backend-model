"""
Evaluation Module
채용공고 품질 평가 모듈
"""

from .data_collector import (
    collect_evaluation_data_async,
    collect_evaluation_data,
    collect_multiple_posts_async,
    collect_multiple_posts
)
from .modules.module_readability import collect_readability_data
from .modules.module_specificity import collect_specificity_data
from .modules.module_attractiveness import collect_attractiveness_data
from .report_saver import (
    save_raw_evaluation_data,
    save_final_report,
    load_evaluation_data
)
from .json_loader import (
    load_evaluation_json,
    list_available_evaluations,
    delete_evaluation_json
)

__all__ = [
    # Data Collector
    "collect_evaluation_data_async",
    "collect_evaluation_data",
    "collect_multiple_posts_async",
    "collect_multiple_posts",
    # Modules
    "collect_readability_data",
    "collect_specificity_data",
    "collect_attractiveness_data",
    # Report Saver
    "save_raw_evaluation_data",
    "save_final_report",
    "load_evaluation_data",
    # JSON Loader
    "load_evaluation_json",
    "list_available_evaluations",
    "delete_evaluation_json",
]

