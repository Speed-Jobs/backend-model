# Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치 (Python 패키지 빌드용)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pip, setuptools, wheel 업그레이드
RUN pip install --upgrade pip setuptools wheel

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright 브라우저 설치 (크롤러용)
RUN playwright install chromium && \
    playwright install-deps chromium

# data 디렉토리 복사 (직무 정의 파일 등)
COPY data/ /app/data/

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 두 프로세스를 안전하게 실행
CMD ["/bin/bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 & python -m app.core.schedular.schedular_skill_model & wait -n; kill $(jobs -p) 2>/dev/null || true"]


#   python -m app.core.schedular.scheduler & 