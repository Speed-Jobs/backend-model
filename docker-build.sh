#!/bin/bash

IMAGE_NAME="speedjobs-backend"
VERSION="1.0.0"
CPU_PLATFORM=amd64
IS_CACHE="--no-cache"

# Docker 이미지 빌드
docker build \
  --tag ${IMAGE_NAME}:${VERSION} \
  --file Dockerfile \
  --platform linux/${CPU_PLATFORM} \
  ${IS_CACHE} .

echo "Docker 이미지 빌드 완료: ${IMAGE_NAME}:${VERSION}"

