# Sentence-BERT Model Service

텍스트 임베딩 및 유사도 계산을 담당하는 독립 마이크로서비스입니다.

## 🎯 목적

- 메인 API 서버와 모델 서빙 로직 분리
- 독립적인 스케일링 및 배포
- 리소스 효율적 운영

## 📦 구조

```
model-service/
├── main.py              # FastAPI 애플리케이션
├── requirements.txt     # Python 의존성
├── Dockerfile          # 컨테이너 이미지
└── README.md           # 이 파일
```

## 🚀 로컬 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 서비스 시작

```bash
python main.py
```

서비스가 `http://localhost:8000`에서 시작됩니다.

### 3. API 문서 확인

브라우저에서 `http://localhost:8000/docs` 접속

## 🐳 Docker 실행

### 1. 이미지 빌드

```bash
docker build -t model-service:latest .
```

### 2. 컨테이너 실행

```bash
docker run -p 8000:8000 model-service:latest
```

## ☸️ Kubernetes 배포

### 1. 이미지 푸시

```bash
docker tag model-service:latest your-registry/model-service:latest
docker push your-registry/model-service:latest
```

### 2. 배포

```bash
kubectl apply -f ../k8s/model-service-deployment.yaml
```

### 3. 서비스 확인

```bash
kubectl get pods -l app=model-service
kubectl logs -f deployment/model-service
```

## 📡 API 엔드포인트

### GET /health
헬스체크

```bash
curl http://localhost:8000/health
```

### POST /embed
텍스트 임베딩 생성

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Python 개발자", "Backend Engineer"],
    "normalize": true
  }'
```

### POST /similarity
유사도 계산

```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Python 백엔드 개발자",
    "corpus_embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
  }'
```

## 🧪 테스트

```bash
# 모델 서비스 시작 후
pytest ../tests/model_service/test_model_service.py -v
```

## 🔧 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MODEL_NAME` | 사용할 Sentence-BERT 모델 | `sentence-transformers/distiluse-base-multilingual-cased-v2` |

## 📊 리소스 요구사항

- **메모리**: 최소 2GB (권장 4GB)
- **CPU**: 최소 1 core (권장 2 cores)
- **디스크**: 약 1GB (모델 캐시)

## 🔍 트러블슈팅

### 모델 로딩 실패
- 메모리 부족: Pod 메모리 제한 확인
- 네트워크 이슈: 모델 다운로드 실패 시 재시도

### 타임아웃 발생
- `initialDelaySeconds` 증가 (모델 로딩 시간)
- `timeout` 값 조정

## 📝 참고사항

- 첫 실행 시 모델 다운로드로 시간 소요 (30초~1분)
- 모델은 메모리에 캐시되어 재사용됨
- 동시 요청 처리 가능 (비동기 FastAPI)

