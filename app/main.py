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
    routers_recruit_counter,
    routers_competitor_recruit_counter,
    routers_competitor_industry_trend,
    routers_skill_insights,
    job_matching,
)
from app.routers.v1.agent import evaluation

app = FastAPI(
    title="Speedjobs Backend API",
    description="채용공고 평가 및 분석 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router)
app.include_router(routers_skill_match.router)
app.include_router(routers_recruit_counter.router)
app.include_router(routers_competitor_recruit_counter.router)
app.include_router(routers_competitor_industry_trend.router)
app.include_router(routers_skill_insights.router)
app.include_router(job_matching.router)
app.include_router(evaluation.router, prefix="/api/v1") 

 

