# Kavier — multi-stage build. Stage 1 builds a wheel with `uv build`; the runtime stages install ONLY
# that wheel (precompiled to bytecode) — no source tree, no PYTHONPATH. Two final targets multiplex the
# two entrypoints off one shared runtime:
#     docker build --target cli -t kavier:cli .   # -> `kavier` unified CLI (default)
#     docker build --target ui  -t kavier:ui  .   # -> `kavier-ui` interactive REPL
# Run:  docker run --rm kavier:cli inference --help
#       docker run --rm -it kavier:ui

# ---- stage 1: build the wheel ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /src
COPY . .
RUN uv build --wheel --out-dir /dist

# ---- shared runtime: install the built wheel, compiled to bytecode ----
FROM python:3.13-slim AS runtime
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir --compile /tmp/*.whl && rm -f /tmp/*.whl

# ---- target: unified CLI (default image) ----
FROM runtime AS cli
ENTRYPOINT ["kavier"]
CMD ["--help"]

# ---- target: interactive REPL ----
FROM runtime AS ui
ENTRYPOINT ["kavier-ui"]
