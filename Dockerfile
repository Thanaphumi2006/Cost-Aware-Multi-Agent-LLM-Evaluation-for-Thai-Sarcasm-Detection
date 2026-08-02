# Thai Sarcasm Detector -- the public demo (Gold/serve_public.py) in a container.
#
#   docker build -t sarcasm-demo .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... sarcasm-demo
#   open http://localhost:8000/app
#
# Only the safe surface is exposed (/app, /api/fetch_comments, /api/escalate, /healthz);
# the key comes from the environment only. Behind a reverse proxy add  -e TRUST_PROXY=1.
#
# WangchanBERTa (the free middle tier) is trained INTO the image by default so the container
# is self-contained and starts instantly. For a lean cue -> GPT image (no torch model, ~1/3
# the size) build with:  docker build --build-arg WITH_WCB=0 -t sarcasm-demo .

FROM python:3.11-slim

# libgomp1 is needed by torch; clean apt lists to keep the layer small
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for better layer caching (requirements are pinned + validated)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# app code (see .dockerignore -- .env, .venv, caches, scraped data are excluded)
COPY . .

# build the WangchanBERTa model into the image (downloads the base model from HF, ~10-15 min).
# skip for a lean image with --build-arg WITH_WCB=0.
ARG WITH_WCB=1
RUN if [ "$WITH_WCB" = "1" ]; then \
        cd Gold && python train_final_wcb.py && rm -rf /root/.cache/huggingface ; \
    else \
        echo "skipping WangchanBERTa -- cascade degrades to cue -> GPT" ; \
    fi

ENV PORT=8000
EXPOSE 8000
WORKDIR /app/Gold

# gunicorn, 2 workers; long timeout so the first (lazy) model load never trips a worker
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT} --timeout 120 serve_public:app"]
