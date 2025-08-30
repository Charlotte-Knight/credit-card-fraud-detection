# /usr/bin/env bash

docker compose down
docker system prune --volumes -f
docker compose up -d --build --force-recreate --remove-orphans

xdg-open http://localhost:8000/docs >/dev/null 2>&1 &
xdg-open http://localhost:3000/d/ccfd/ccfd >/dev/null 2>&1 &

tmux new-session -d -s monitoring watch -n 5 -t 'docker ps --format "table {{.Names}}\t{{.Status}}"'
tmux split-window -h 'docker logs -f fastapi'
tmux split-window -v 'docker logs -f synthetic_data'
tmux set-hook client-resized 'resize-pane -t 0 -x 41'
tmux attach -d