#!/bin/bash
set -e

cd /home/michael_lin/playground/text-embeddings-inference

TAG=$(date +%Y%m%d%H%M)
echo "=== Build Tag: ${TAG} ===" | tee /tmp/build_tag.txt

PROXY_ARGS="--build-arg http_proxy=http://proxy.intra:80 --build-arg https_proxy=http://proxy.intra:80"

echo ""
echo "=== Building CPU version: tei-bge-m3-cpu:${TAG} ===" 
DOCKER_BUILDKIT=1 docker build ${PROXY_ARGS} -t tei-bge-m3-cpu:${TAG} -f Dockerfile-cpu-python .
echo "✅ CPU build completed!"

echo ""
echo "=== Building GPU version: tei-bge-m3-gpu:${TAG} ===" 
DOCKER_BUILDKIT=1 docker build ${PROXY_ARGS} -t tei-bge-m3-gpu:${TAG} -f Dockerfile-cuda-python .
echo "✅ GPU build completed!"

echo ""
echo "=== Build Summary ==="
docker images | grep "tei-bge-m3"
echo ""
echo "CPU Image: tei-bge-m3-cpu:${TAG}"
echo "GPU Image: tei-bge-m3-gpu:${TAG}"
