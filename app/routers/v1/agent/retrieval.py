from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.retrieval_service import RetrievalService
from app.services.agentic_rag_service import AgenticRAGService
from app.schemas.retrieval import SearchQuery, SearchResult, SearchResponse, AgenticRAGQuery, AgenticRAGResponse
from app.core.database import get_db
from app.models.company import Company

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("/", response_model=SearchResponse)
async def search(query: SearchQuery, db: Session = Depends(get_db)):
    """
    VectorDB 검색

    - 쿼리 텍스트를 embedding하여 유사한 문서 검색
    - 회사명으로 필터링 가능 (예: "카카오", "toss")
    - 회사명 미입력시 전체에서 검색

    Example:
    ```json
    {
        "text": "백엔드 개발자 채용",
        "top_k": 5,
        "company_name": "카카오"
    }
    ```
    """
    try:
        # 필터 준비
        filters = query.filters or {}

        # 회사명으로 company_id 조회
        if query.company_name:
            company = db.query(Company).filter(
                Company.name.ilike(f"%{query.company_name}%"),
                Company.is_deleted == False
            ).first()

            if company:
                filters["company_id"] = company.id
            else:
                # 회사를 찾지 못한 경우 빈 결과 반환
                return SearchResponse(
                    query=query.text,
                    results=[],
                    total_results=0
                )

        # 검색 실행
        service = RetrievalService()
        results = await service.search(
            query=query.text,
            top_k=query.top_k,
            filters=filters if filters else None,
            db=db
        )

        return SearchResponse(
            query=query.text,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test")
async def test_search(q: str = "개발자", top_k: int = 3):
    """
    간단한 테스트 검색 엔드포인트

    - GET 방식으로 빠르게 테스트 가능
    """
    try:
        service = RetrievalService()
        results = await service.search(query=q, top_k=top_k)

        return SearchResponse(
            query=q,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agentic", response_model=AgenticRAGResponse)
async def agentic_search(query: AgenticRAGQuery, db: Session = Depends(get_db)):
    """
    Agentic RAG 검색 (LangGraph + GPT-4o + Tavily + SQL Analytics)

    ## 지능형 자동 처리
    LLM이 사용자 질문만으로 모든 것을 자동 판단:
    - **엔티티 추출**: 회사명, 날짜, 직무, 고용형태 등 자동 인식
    - **라우팅 결정**: vectordb / websearch / both / statistics_with_stats 자동 선택
    - **결과 개수**: 질문 유형에 맞는 top_k 자동 결정
    - **통계 분석**: "몇 개?", "평균", "트렌드" 등의 질문 시 SQL 통계 분석 자동 실행

    ## 검색 소스
    1. **vectordb**: 특정 회사/직무 채용 공고 검색
    2. **websearch**: 실시간 웹 검색 (최신 트렌드, 뉴스)
    3. **both**: 채용 공고 + 웹 정보 결합
    4. **statistics_with_stats**: 채용 공고 통계 분석 (DB 집계 쿼리 실행)

    ## 사용 예시
    ```json
    {
        "text": "2025년 하반기 토스 채용공고 총 몇개야?"
    }
    ```
    → LLM이 자동으로:
    - 회사명 "토스" 추출
    - 통계 쿼리 필요 판단
    - SQL 생성 및 실행
    - 결과 기반 인사이트 생성

    ```json
    {
        "text": "카카오 백엔드 개발자 채용 요구사항"
    }
    ```
    → vectordb 검색 + 답변 생성

    ```json
    {
        "text": "2025년 AI 개발 트렌드"
    }
    ```
    → websearch로 최신 정보 검색

    ## Response
    - **query**: 입력 질문
    - **answer**: LLM이 생성한 답변 (통계 + 인사이트)
    - **sources**: 검색된 소스 문서들 (vectordb + web)
    - **route_decision**: 라우팅 결정
    - **total_sources**: 전체 소스 개수
    """
    try:
        service = AgenticRAGService()
        result = await service.search(
            query=query.text,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
