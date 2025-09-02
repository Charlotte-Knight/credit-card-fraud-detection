FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

COPY ./app ./app

HEALTHCHECK --start-period=5s --start-interval=10s --interval=1s --timeout=2s --retries=3 CMD curl --fail http://localhost:8000/health || exit 1

CMD ["fastapi", "run", "app/main.py"]