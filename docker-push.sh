#!/bin/bash

IMAGE_NAME="skala-backend-model"
VERSION="1.0.0"
DOCKER_HUB_USERNAME="progamm3r"

# 환경 변수에서 비밀번호 읽기
if [ -z "$DOCKER_HUB_PASSWORD" ]; then
    echo "Error: DOCKER_HUB_PASSWORD 환경 변수가 설정되지 않았습니다."
    echo "사용법: export DOCKER_HUB_PASSWORD='your_token' && ./docker-push.sh"
    exit 1
fi

# 1. Docker Hub에 로그인
echo ${DOCKER_HUB_PASSWORD} | docker login \
    -u ${DOCKER_HUB_USERNAME} --password-stdin \
    || { echo "Docker 로그인 실패"; exit 1; }

# 2. Docker Hub용 태그 추가
docker tag ${IMAGE_NAME}:${VERSION} ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}

# 3. Docker 이미지 푸시
docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}

echo "Docker 이미지 푸시 완료: ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}"