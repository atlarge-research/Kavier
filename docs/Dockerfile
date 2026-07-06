# Documentation-only image: builds and serves the Kavier MkDocs site.
# Build context is this docs/ directory:
#     docker build -t kavier-docs docs/
#     docker run --rm -p 8000:8000 kavier-docs      # live-reload dev server on :8000
#
# To emit the static site instead of serving it:
#     docker run --rm -v "$PWD/site:/docs/site" kavier-docs mkdocs build
FROM python:3.13-slim

WORKDIR /docs

# Install docs deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MkDocs project (mkdocs.yml, content/, javascripts/, usage.py, ...).
COPY . .

EXPOSE 8000

# Serve with live reload, bound to all interfaces so the port is reachable.
CMD ["mkdocs", "serve", "--dev-addr", "0.0.0.0:8000"]
