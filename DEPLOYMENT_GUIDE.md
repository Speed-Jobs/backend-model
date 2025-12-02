# 모델 서비스 분리 배포 가이드

## 📋 개요

Sentence-BERT 모델을 별도 마이크로서비스로 분리하여 배포합니다.

### 변경 사항
- **기존**: 메인 API에 모델 포함 (무거움)
- **변경**: 모델 서비스 분리 (가벼움)

### 장점
- ✅ 메인 API 배포 속도 향상 (5분 → 1분)
- ✅ 독립적 스케일링 (API 10개, 모델 2개)
- ✅ 리소스 효율 (메모리 50% 절약)
- ✅ 장애 격리 (모델 문제가 전체 API에 영향 없음)

---

## 🚀 배포 순서

### Phase 1: 모델 서비스 배포 (먼저!)

```bash
# 1. 이미지 빌드
cd model-service
docker build -t your-registry/model-service:v1.0 .

# 2. 이미지 푸시
docker push your-registry/model-service:v1.0

# 3. Kubernetes 배포
kubectl apply -f ../k8s/model-service-deployment.yaml

# 4. 배포 확인
kubectl get pods -l app=model-service
kubectl logs -f deployment/model-service

# 5. 헬스체크 확인
kubectl port-forward svc/model-service 8001:8000
curl http://localhost:8001/health
```

**중요**: 모델 서비스가 완전히 시작될 때까지 대기 (약 1~2분)

### Phase 2: 메인 API 배포

```bash
# 1. k8s/api-deployment.yaml 수정
# image: your-registry/api-service:v1.0  # 실제 레지스트리 주소

# 2. 환경 변수 확인
# MODEL_SERVICE_URL: "http://model-service:8000"

# 3. 이미지 빌드 (모델 없이 가벼움!)
docker build -t your-registry/api-service:v1.0 .

# 4. 이미지 푸시
docker push your-registry/api-service:v1.0

# 5. Kubernetes 배포
kubectl apply -f k8s/api-deployment.yaml

# 6. 배포 확인
kubectl get pods -l app=api-service
kubectl logs -f deployment/api-service
```

---

## 🧪 로컬 테스트 (Docker Compose)

### 1. 모델 서비스만 테스트

```bash
# 1. 서비스 시작
docker-compose -f docker-compose.test.yml up model-service

# 2. 헬스체크
curl http://localhost:8001/health

# 3. 임베딩 테스트
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Python 개발자"], "normalize": true}'
```

### 2. 통합 테스트

```bash
# 1. 전체 서비스 시작
docker-compose -f docker-compose.test.yml up

# 2. pytest 실행
pytest tests/model_service/test_model_service.py -v
```

---

## 🔍 배포 검증

### 1. 모델 서비스 확인

```bash
# Pod 상태
kubectl get pods -l app=model-service

# 로그 확인
kubectl logs -f deployment/model-service

# 헬스체크
kubectl exec -it deployment/model-service -- curl http://localhost:8000/health
```

### 2. 메인 API 확인

```bash
# Pod 상태
kubectl get pods -l app=api-service

# 로그에서 모델 서비스 연결 확인
kubectl logs -f deployment/api-service | grep "ModelServiceClient"

# 직무 매칭 테스트
curl -X POST http://your-api-url/api/job-matching/match \
  -H "Content-Type: application/json" \
  -d '{"post_id": 123}'
```

### 3. 통신 확인

```bash
# API Pod에서 모델 서비스 호출 테스트
kubectl exec -it deployment/api-service -- curl http://model-service:8000/health
```

---

## 📊 모니터링

### 리소스 사용량

```bash
# 모델 서비스
kubectl top pods -l app=model-service

# 메인 API
kubectl top pods -l app=api-service
```

### 로그 모니터링

```bash
# 모델 서비스 로그
kubectl logs -f deployment/model-service --tail=100

# API 서비스 로그
kubectl logs -f deployment/api-service --tail=100
```

---

## 🔧 트러블슈팅

### 문제 1: 모델 서비스 시작 실패

**증상**: Pod가 CrashLoopBackOff 상태

**원인**: 메모리 부족

**해결**:
```yaml
# k8s/model-service-deployment.yaml
resources:
  requests:
    memory: "4Gi"  # 2Gi → 4Gi로 증가
```

### 문제 2: API에서 모델 서비스 연결 실패

**증상**: `ModelServiceClient 초기화 실패`

**원인**: 모델 서비스가 아직 시작 안 됨

**해결**:
```bash
# 모델 서비스 상태 확인
kubectl get pods -l app=model-service

# 모델 서비스가 Ready 상태가 될 때까지 대기
kubectl wait --for=condition=ready pod -l app=model-service --timeout=300s

# 그 후 API 재시작
kubectl rollout restart deployment/api-service
```

### 문제 3: 타임아웃 발생

**증상**: `모델 서비스 타임아웃 (30초)`

**원인**: 대량 텍스트 처리 시 시간 초과

**해결**:
```python
# app/utils/model_service_client.py
client = ModelServiceClient(timeout=60)  # 30초 → 60초
```

### 문제 4: 파이프라인 실패

**증상**: 3일마다 실행되는 파이프라인에서 직무 매칭 실패

**원인**: 모델 서비스 다운

**해결**:
```bash
# 모델 서비스 상태 확인
kubectl get pods -l app=model-service

# 재시작
kubectl rollout restart deployment/model-service

# 파이프라인 재실행
python app/scripts/pipeline/data_pipeline.py
```

---

## 🎯 롤백 방법

문제 발생 시 기존 방식으로 롤백:

```python
# app/core/job_matching/job_matching_system.py
# 53번째 줄 주석 처리
# from app.utils.model import ModelServiceClient as SentenceTransformer
from sentence_transformers import SentenceTransformer  # 기존 방식
```

---

## 📈 성능 비교

| 항목 | 기존 | 분리 후 |
|------|------|---------|
| API 이미지 크기 | 2GB | 500MB |
| API 시작 시간 | 60초 | 10초 |
| API 메모리 사용 | 2GB/Pod | 512MB/Pod |
| 배포 시간 | 5분 | 1분 |
| 응답 시간 | 200ms | 210~250ms |

---

## 🔗 관련 문서

- [모델 서비스 API 문서](http://model-service:8000/docs)
- [메인 API 문서](http://api-service:8000/docs)

