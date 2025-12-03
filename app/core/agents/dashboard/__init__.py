"""
Dashboard 인사이트 Agent
특정 회사의 채용 공고 수 추이와 경쟁사 비교 분석을 제공하는 Agent
"""

from app.core.agents.dashboard.recruitment_trends.company_insight_agent import (
    generate_company_insight_async,
)
from app.core.agents.dashboard.job_role.job_role_insight_agent import (
    generate_job_role_insight_async,
)

__all__ = [
    "generate_company_insight_async",
    "generate_job_role_insight_async",
]

