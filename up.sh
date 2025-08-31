# /usr/bin/env bash

docker compose down
docker system prune --volumes -f
docker compose up -d --build --force-recreate --remove-orphans

xdg-open http://localhost:8000/docs >/dev/null 2>&1 &
xdg-open http://localhost:3000/d/ccfd/ccfd >/dev/null 2>&1 &

tmux new-session -d -s monitoring watch -n 5 -t 'docker ps --format "table {{.Names}}\t{{.Status}}"'
tmux split-window -h 'docker logs -f ccfd_api'
tmux split-window -v 'docker logs -f transaction_generator'
tmux set-hook client-resized 'resize-pane -t 0 -x 50'
tmux attach -d