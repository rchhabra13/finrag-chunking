FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY config.yaml ./

# Local LLM servers run on the host; reach them via host.docker.internal, e.g.:
#   docker run --add-host=host.docker.internal:host-gateway \
#     -v $PWD/data:/app/data -v $PWD/results:/app/results finrag \
#     eval --model ollama/llama3.2:3b
# (override endpoint base_urls in config.yaml accordingly)
ENTRYPOINT ["uv", "run", "finrag"]
CMD ["--help"]
