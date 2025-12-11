from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings

# 기존 SQLAlchemy 모델 import (relationship 등 ORM Initializing을 위해 필요)
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

# 기존 라우터들 import
from app.routers import (
    routers_skill_match,
    user,
    routers_recruit_counter,
    routers_competitor_recruit_counter,
    routers_competitor_industry_trend,
    routers_skill_insights,
    job_matching,
    routers_recruitment_schedule,
    rag_retrieval,
)
from app.routers.v1.agent import evaluation

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME if hasattr(settings, "APP_NAME") else "Speedjobs Backend API",
    description=getattr(settings, "APP_DESCRIPTION", "채용공고 평가 및 분석 API"),
    version=settings.APP_VERSION if hasattr(settings, "APP_VERSION") else "1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 변경 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기존 라우터 등록
app.include_router(user.router)
app.include_router(routers_skill_match.router)
app.include_router(routers_recruit_counter.router)
app.include_router(routers_competitor_recruit_counter.router)
app.include_router(routers_competitor_industry_trend.router)
app.include_router(routers_skill_insights.router)
app.include_router(job_matching.router)
app.include_router(routers_recruitment_schedule.router)
app.include_router(evaluation.router, prefix="/api/v1") 

# RAG 시스템 관련 라우터 추가 등록
app.include_router(rag_retrieval.router)

@app.get("/")
async def root():
    """API 루트 엔드포인트 - 서비스 소개 및 기본 정보 제공"""
    return {
        "message": "Welcome to SpeedJobs API - Backend & VectorDB RAG Endpoints",
        "backend_version": getattr(settings, "APP_VERSION", "1.0.0"),
        "vector_api_docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7777,
        reload=getattr(settings, "DEBUG", False)
    )
