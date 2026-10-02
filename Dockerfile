# Build a wheel with uv, then install only the wheel in a slim runtime image.
#     docker build -t kavier .
#     docker run --rm kavier inference --help

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

FROM python:3.13-slim
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir --compile /tmp/*.whl && rm -f /tmp/*.whl
ENTRYPOINT ["kavier"]
CMD ["--help"]
