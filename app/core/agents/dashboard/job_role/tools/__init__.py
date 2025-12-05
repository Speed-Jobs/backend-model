"""
Job Role Insight Agent Tools
"""

from app.core.agents.dashboard.job_role.tools.data_fetcher import (
    get_job_role_statistics_data,
)
from app.core.agents.dashboard.job_role.tools.news_fetcher import (
    search_job_role_news,
)

__all__ = [
    "get_job_role_statistics_data",
    "search_job_role_news",
]

