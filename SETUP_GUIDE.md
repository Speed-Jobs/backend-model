# Backend Model + RAG Setup Guide

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
cd backend-model
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=your_password
DB_NAME=speedjobs

# AI/LLM API Keys (RAG 시스템용)
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

# Debug Mode (optional)
DEBUG=False
```

### 3. 서버 실행

```bash
python -m app.main
```

또는

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7777 --reload
```

서버는 `http://localhost:7777`에서 실행됩니다.

## 📚 API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: `http://localhost:7777/docs`
- ReDoc: `http://localhost:7777/redoc`

## 🤖 RAG 엔드포인트 사용

### RAG 검색

```bash
POST http://localhost:7777/rag/search

{
  "text": "2025년 하반기 토스 채용공고 총 몇개야?"
}
```

### Health Check

```bash
GET http://localhost:7777/rag/health
```

## ⚠️ 문제 해결

### ModuleNotFoundError: No module named 'mysql'

```bash
pip install mysql-connector-python
```

### RAG 관련 의존성 오류

```bash
pip install langgraph tavily-python pymysql
```

### VectorSearchTool 관련 오류

현재 VectorSearchTool은 placeholder 구현입니다.
실제 VectorDB를 사용하려면:

1. Qdrant 클라이언트 설치: `pip install qdrant-client`
2. `app/core/agents/rag/tools/vector_search.py` 파일의 TODO 부분 구현

자세한 내용은 [RAG_MIGRATION_GUIDE.md](RAG_MIGRATION_GUIDE.md)를 참고하세요.

## 📖 추가 문서

- [RAG_MIGRATION_GUIDE.md](RAG_MIGRATION_GUIDE.md) - RAG 시스템 마이그레이션 가이드
- [vectorDB-server/AGENT_ARCHITECTURE.md](../vectorDB-server/AGENT_ARCHITECTURE.md) - Agent 아키텍처 상세 설명
