# Releasing kavier to PyPI

Publishes via GitHub Actions **Trusted Publishing** (OIDC — no API token to manage).

## One-time PyPI setup (human; required before the first release)

1. Create/own a PyPI account or organisation for `atlarge-research`.
2. PyPI → *Your account → Publishing* → **Add a pending publisher**:
   - PyPI Project Name: `kavier`
   - Owner: `atlarge-research` · Repository: `Kavier` · Workflow: `publish.yml`
   - Environment: *(leave blank)*

   This claims the `kavier` name and authorises this repo to publish.

## Each release

1. Merge the release PR into `master`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`
3. `.github/workflows/publish.yml` builds the sdist + wheel and uploads to PyPI automatically.

## Manual fallback (API token, instead of the workflow)

```bash
python -m build && twine check dist/*
TWINE_USERNAME=__token__ TWINE_PASSWORD=<pypi-token> twine upload dist/*
```
