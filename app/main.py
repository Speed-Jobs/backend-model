from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# SQLAlchemy 모델 import (관계 설정을 위해 필요)
from app.models import (
    company,
    industry,
    post,
    skill,
    post_skill,
    position,
    position_skill,
    industry_skill,
    dashboard_stat,
)

from app.routers import (
    routers_skill_match,
    user,
    routers_competitors_skills,
    routers_recruit_counter,
    routers_competitor_recruit_counter,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
)

app.include_router(user.router)
app.include_router(routers_skill_match.router)
app.include_router(routers_competitors_skills.router)
app.include_router(routers_recruit_counter.router)
app.include_router(routers_competitor_recruit_counter.router)