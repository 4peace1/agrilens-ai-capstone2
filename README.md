# AgriLens AI — Decentralized Crop Diagnostics Pipeline

[![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/<repo>/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Capstone implementation for the AgriLens AI brief: an automated, low-latency
classification pipeline that replaces the 48-72 hour manual agronomist
review with sub-2-second, offline-tolerant disease diagnosis for cassava,
cocoa, and maize.

> Replace `<org>/<repo>` in the badge URL above with your actual GitHub
> path once pushed, or delete the badge line if you'd rather not show it
> until the first CI run completes.


## Architecture

```
Mobile App (offline-first)
      │ 1. POST /api/v1/diagnostics  (metadata only)
      ▼
FastAPI Gateway ──── PostgreSQL + PostGIS (status, geo-indexed location)
      │ 2. signed GCS PUT URL returned
      ▼
GCS raw/ bucket  ◄── mobile app uploads image bytes directly (gateway never touches them)
      │ 3. POST /api/v1/diagnostics/{id}/complete
      ▼
FastAPI Gateway ──background task──► Pub/Sub (image-processing-topic)
                                            │
                                            ▼
                              Image Processing Worker (GKE, autoscaled
                              on Pub/Sub backlog)
                                  • streamed download, capped memory
                                  • resize + letterbox pad to 224x224
                                  • upload to GCS processed/ bucket
                                            │
                                            ▼
                          Inference Orchestrator (TFLite, GKE)
                            ┌─────────────┴─────────────┐
                     Cassava Model Pods            Cocoa/Maize Model Pods
                     (A/B router: stable            (A/B router: stable
                      vs canary version)              version)
                                            │
                                            ▼
                          PostgreSQL (result persisted) + Pub/Sub
                          (inference-results-topic)
                                            │
                                            ▼
                              Notification Service ── FCM push ──► Mobile App
```

## Repository layout

```
app/                  FastAPI gateway (Phase 1)
  main.py             HTTP endpoints, lifecycle, background-task dispatch
  config.py           Env-driven settings (12-factor)
  schemas.py          Pydantic models + DiagnosticStatus state machine
  db.py               PostGIS-backed persistence, idempotent status transitions
  storage.py           Signed GCS URL generation
  pubsub_client.py     Pub/Sub publisher + circuit breaker
  outbox.py            Transactional outbox (circuit-breaker fallback)
  privacy.py            Geospatial masking (Phase 3)

worker/                Image-processing worker (Phase 1)
  image_processor.py    Streamed download, resize/pad, idempotent consumer

inference/              Inference layer (Phase 2)
  client.py              Worker → orchestrator HTTP client (2.0s SLA timeout)
  quantize.py             PyTorch → ONNX → TFLite int8 quantization pipeline
  model_server/
    app.py               TFLite-serving FastAPI app, per-crop model loading
    ab_router.py          Deterministic A/B / canary routing

training/                Model training pipeline (feeds Phase 2)
  synthetic_data.py       Procedural smoke-test data, zero downloads
  dataset.py               ImageFolder loading, augmentation, class weighting
  train.py                  ResNet18 transfer-learning training loop
  evaluate.py                Held-out test set evaluation vs. the F1 floor
  README.md                  Verified public dataset links + workflow

notification/           Notification Service — Pub/Sub → FCM push
scripts/
  reconcile_outbox.py    Outbox reconciler (run as a CronJob)

k8s/                     Kubernetes manifests
  gateway-deployment.yaml         Gateway Deployment + Service + HPA (CPU)
  worker-deployment.yaml          Worker Deployment + HPA (Pub/Sub backlog)
  inference-deployment.yaml       Per-crop inference Deployments + HPAs
  configmap-secrets.yaml          Shared env config
  outbox-reconciler-cronjob.yaml  Outbox reconciler CronJob

tests/                   pytest suite (pure logic + mocked-API tests)
docker-compose.yml        Local dev stack (Postgres+PostGIS, Pub/Sub &
                           GCS emulators, gateway, worker, inference)

.github/workflows/ci.yml  Lint, test, k8s-manifest validation, Docker builds
Makefile                  make install-dev / lint / test / up / down / build
pyproject.toml             ruff lint config
CONTRIBUTING.md            Dev setup + PR checklist
LICENSE                    MIT
```

## How each phase maps to the brief

### Phase 1 — Decoupled Pre-processing & Asynchronous Ingestion
- `POST /api/v1/diagnostics` validates metadata and returns a **signed GCS
  upload URL** in milliseconds — the gateway never buffers image bytes.
- `POST /api/v1/diagnostics/{id}/complete` confirms the upload and
  publishes to **Pub/Sub via a `BackgroundTask`**, so the HTTP response
  returns immediately and heavy lifting happens entirely off the request
  path (`app/main.py::_dispatch_processing`).
- The worker streams images from GCS in bounded chunks and uses
  `Image.draft()` to downscale *during* JPEG decode — the direct fix for
  the OOM root cause — plus a Pub/Sub `FlowControl(max_messages=4)` cap
  that bounds peak in-flight memory per pod.
- Coordinates are validated and stored as `GEOGRAPHY(Point, 4326)` with a
  GIST index, not naive floats (`app/db.py`).

### Phase 2 — Optimized Inference Layer with TFLite & Triton
- `training/` is the model-training pipeline that *feeds* this phase —
  see `training/README.md` for the three verified public datasets
  (Kaggle cassava-leaf-disease-classification, corn-or-maize-leaf-
  disease-dataset, and a cacao disease dataset) plus a synthetic-data
  mode for smoke-testing the whole pipeline without any downloads.
  `training/train.py` does ResNet18 transfer learning with class-
  weighted loss (the real cassava dataset is ~62% one class) and tracks
  macro-F1 every epoch against the brief's 85% floor.
- `inference/quantize.py` documents the PyTorch → ONNX → TFLite **int8
  post-training quantization** pipeline (the brief's "Quantize your
  PyTorch models to TFLite" guidance), with representative-dataset
  calibration so F1 doesn't silently degrade. It also carries the
  trained model's `labels.json` sidecar through to the served artifact,
  so `model_server/app.py` always serves the exact class list a model
  was actually trained on instead of a hand-maintained list that can
  drift out of sync.
- `inference/model_server/app.py` loads TFLite interpreters **once at
  startup**, runs CPU-only inference, and is deployed as its **own GKE
  Deployment** (`k8s/inference-deployment.yaml`) — separate from the
  gateway, per "Don't use a single large container for both the API and
  the ML model."
- Cassava and Cocoa run as **separate Deployments** (`CROP_FILTER` env
  var) with independent HPAs, so a cassava traffic burst during planting
  season scales without starving cocoa capacity.
- `inference/model_server/ab_router.py` does deterministic, sticky A/B
  routing between a stable and canary model version per crop.
- `inference/client.py` enforces the **2.0s hard SLA** as the actual HTTP
  timeout, failing fast rather than letting a slow model create
  backpressure into the Pub/Sub queue.

### Phase 3 — Geospatial Data Integrity & Privacy Shield
- Exact coordinates are stored once (for legitimate internal spatial
  queries) but **never returned across the API boundary** — every
  response is run through `app/privacy.py::mask_coordinates`, which
  snaps to a coarse grid and applies a deterministic per-diagnostic
  jitter so the true farm location can't be reconstructed by repeated
  querying.
- `captured_at` is a **client-supplied timestamp**, honoring the
  offline-first constraint: a farmer can capture a photo with zero
  signal and sync hours later without losing the true capture time.

### Resilience & idempotency (cuts across all phases)
- **Idempotent consumer pattern**: every DB status transition
  (`mark_queued`, `mark_processing`, `mark_completed`) is a conditional
  `UPDATE ... WHERE status = ...`, so Pub/Sub's at-least-once redelivery
  can never double-process or corrupt state (`app/db.py`,
  `worker/image_processor.py::process_message`).
- **Circuit breaker**: `app/pubsub_client.py` trips after repeated
  publish failures and fails fast rather than hanging the
  farmer-facing `/complete` call; failed publishes durably land in a
  Postgres **outbox** (`app/outbox.py`) replayed by
  `scripts/reconcile_outbox.py` (deployed as a 1-minute CronJob).
- **Correlation IDs** thread through every log line from gateway →
  Pub/Sub → worker → inference, for end-to-end tracing of a single
  farmer's image.

### Observability (bonus)
- `prometheus-fastapi-instrumentator` exposes `/metrics` on the gateway
  with zero extra code — request latency histograms, in-flight counts,
  status-code breakdowns out of the box.
- Structured logs include `diagnostic_id`, `correlation_id`, model
  version, and per-stage latency at every hop, ready to ship to Cloud
  Logging / a Grafana dashboard.

## Running locally

```bash
cp .env.example .env
docker compose up --build
# Gateway:    http://localhost:8000/docs
# Inference:  http://localhost:8080/docs
# Pub/Sub emulator: localhost:8085
# Fake GCS:   localhost:4443
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

21 tests cover coordinate validation, deterministic privacy masking, A/B
routing distribution, circuit-breaker state transitions, and the gateway
HTTP contract (signed-URL issuance, 422 on bad coordinates, 404 on
unknown diagnostics, masked-location responses, completed-result
serialization) — all without requiring live GCP infra. True
end-to-end integration tests would run against the emulator stack in
`docker-compose.yml`.

## Producing real model artifacts

The `.tflite` files referenced in `inference/model_server/app.py` are
**not bundled** — see `training/README.md` for the full workflow
(verified public datasets, training, evaluation against the F1 floor)
and `inference/quantize.py` for the conversion step:

```bash
python -m training.train --crop cassava --data-dir data/cassava \
    --output checkpoints/cassava_resnet18.pt --epochs 15

python -m inference.quantize \
    --checkpoint checkpoints/cassava_resnet18.pt \
    --calibration-dir data/calibration/cassava \
    --output models/cassava_v3_int8.tflite
```

The inference service logs a warning and continues booting if an
artifact is missing, so the rest of the stack remains runnable for
local development and code review without requiring trained weights.
The training pipeline (synthetic-data mode) was run end-to-end while
building this repo to confirm the data → train → label-sidecar →
quantize → serve chain is correctly wired; see `training/README.md` for
that run's output.

## Getting this onto GitHub

```bash
cd agrilens-ai-capstone
git init
git add .
git commit -m "Initial commit: AgriLens AI capstone pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

CI (`.github/workflows/ci.yml`) runs automatically on push: ruff lint,
pytest, k8s manifest YAML validation, and a build check for all four
Docker images. No secrets are required for CI to pass — it never touches
real GCP infra.

**Before pushing**, double check `.env` is not tracked (it's git-ignored
by default; only `.env.example` should be committed) and that
`k8s/configmap-secrets.yaml` still has placeholder values, not real
credentials.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local dev setup, the
lint/test workflow, and guidance on adding a new crop or model variant.

## License

[MIT](LICENSE) — see the LICENSE file for details.

## What's intentionally out of scope for this capstone

- The React Native mobile app itself (brief only required the backend
  pipeline; the gateway's signed-URL contract is what the app would
  call).
- Actually downloading and training on the full real datasets — the
  `training/` pipeline is real and runnable, and was smoke-tested
  end-to-end on synthetic data, but the multi-GB real downloads + full
  15-epoch training runs weren't executed as part of building this repo.
  See `training/README.md` for the exact commands to do that on your
  own machine.
- A production secret-management integration (Secret Manager + External
  Secrets Operator) — `k8s/configmap-secrets.yaml` documents the shape
  but uses placeholder values.
- A full Triton Inference Server deployment — `tflite-runtime` was
  chosen for the reference implementation since it directly meets the
  CPU-only, sub-2s, edge-friendly constraint without GPU infrastructure;
  swapping in Triton for the larger maize model is a drop-in change at
  the `inference/client.py` boundary if needed later.
