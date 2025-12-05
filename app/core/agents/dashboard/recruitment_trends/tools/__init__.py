"""
Dashboard Agent Tools
"""

from app.core.agents.dashboard.recruitment_trends.tools.data_fetcher import (
    get_company_recruitment_data,
    get_competitors_recruitment_data,
    get_total_recruitment_data,
)
from app.core.agents.dashboard.recruitment_trends.tools.news_fetcher import (
    search_naver_news,
)

__all__ = [
    "get_company_recruitment_data",
    "get_competitors_recruitment_data",
    "get_total_recruitment_data",
    "search_naver_news",
]
