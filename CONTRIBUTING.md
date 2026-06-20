# Contributing

## Development setup

```bash
git clone <repo-url>
cd agrilens-ai-capstone
cp .env.example .env
make install-dev
```

## Local stack

```bash
make up      # starts Postgres+PostGIS, Pub/Sub & GCS emulators, gateway, worker, inference
make down    # tears it down
```

Gateway docs: http://localhost:8000/docs
Inference docs: http://localhost:8080/docs

## Before opening a PR

```bash
make lint     # ruff
make test     # pytest
```

Both run in CI (`.github/workflows/ci.yml`) on every push and PR — please
make sure they pass locally first.

## Code style

- Formatting/linting is enforced via `ruff` (config in `pyproject.toml`).
- Type hints are expected on public function signatures.
- New endpoints, status transitions, or Pub/Sub message shapes should
  come with a corresponding test in `tests/`.

## Adding a new crop / model

1. Add the crop to `CropType` in `app/schemas.py`.
2. Add a routing entry in `inference/model_server/ab_router.py`'s
   `ROUTING_TABLE`.
3. Add labels to `CLASS_LABELS` in `inference/model_server/app.py`.
4. Produce a quantized model via `inference/quantize.py` and drop it in
   `models/`.
5. If the new crop needs its own scaling profile, give it its own
   Deployment/HPA in `k8s/inference-deployment.yaml` (following the
   cassava/cocoa split already there) rather than overloading an
   existing pod.

## Commit messages

Conventional, short, imperative mood (`fix: handle null-island GPS
fixes`, `feat: add maize model routing`) is preferred but not enforced.
