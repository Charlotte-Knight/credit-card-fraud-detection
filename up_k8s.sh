# /usr/bin/env bash

export GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

mkdir -p k8s/
rm k8s/*

kompose convert -o k8s/

minikube delete
minikube start --driver=docker --cpus="no-limit"
minikube addons enable metrics-server

kubectl apply -f k8s/

minikube service grafana &> /dev/null &

tmux new-session -d -s monitoring watch -n 5 -t 'kubectl get pods'
tmux split-window -h 'stern . default'
tmux attach -d
