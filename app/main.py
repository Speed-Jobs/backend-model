from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import user, job_matching

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
)

app.include_router(user.router)
app.include_router(job_matching.router)