# Sentence-BERT Model Service Dockerfile
# 멀티스테이지 빌드로 이미지 크기 최적화

# Stage 1: 베이스 이미지 (Python + 의존성)
FROM python:3.11-slim as base

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (빌드 도구)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: 런타임 이미지 (불필요한 빌드 도구 제외)
FROM python:3.11-slim

WORKDIR /app

# Stage 1에서 설치한 패키지 복사
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# curl 설치 (헬스체크용)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 애플리케이션 코드 복사
COPY main.py .

# 모델 캐시 디렉토리 생성
RUN mkdir -p /root/.cache/torch/sentence_transformers

# 환경 변수 설정
ENV MODEL_NAME="sentence-transformers/distiluse-base-multilingual-cased-v2"
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
# 모델 로딩 시간을 고려하여 start-period를 60초로 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

