from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# SQLAlchemy 모델들을 import하여 relationship이 제대로 작동하도록 함
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
    job_matching
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
app.include_router(job_matching.router)
