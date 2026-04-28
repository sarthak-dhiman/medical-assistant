#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10000}"
UVICORN_PORT="${UVICORN_PORT:-8000}"
LIGHT_CONCURRENCY="${LIGHT_CONCURRENCY:-2}"
HEAVY_CONCURRENCY="${HEAVY_CONCURRENCY:-1}"

cat > /etc/nginx/conf.d/default.conf <<NGINX
server {
    listen ${PORT};
    client_max_body_size 50M;

    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files \$uri \$uri/ /index.html;
    }

    location /predict {
        proxy_pass http://127.0.0.1:${UVICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /result {
        proxy_pass http://127.0.0.1:${UVICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:${UVICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX

shutdown() {
    jobs -p | xargs -r kill
}
trap shutdown EXIT INT TERM

redis-server --save "" --appendonly no &

celery -A web_app.backend.celery_app worker \
    --loglevel=info \
    --pool=threads \
    --concurrency="${HEAVY_CONCURRENCY}" \
    -Q q_heavy_cv,celery &

celery -A web_app.backend.celery_app worker \
    --loglevel=info \
    --pool=threads \
    --concurrency="${LIGHT_CONCURRENCY}" \
    -Q q_lightweight &

uvicorn web_app.backend.main:app --host 127.0.0.1 --port "${UVICORN_PORT}" &

nginx -g "daemon off;" &

wait -n
