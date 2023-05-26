#!/bin/sh

sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt install -y docker.io
sudo docker run hello-world

echo "test-passed: docker installed successfully"

sudo docker image build -t inference-image .
sudo docker container run -p 8080:8080 -n inference-container inference-image

