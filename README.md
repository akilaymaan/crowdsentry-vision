# CrowdSentry

AI-based **Spatio-Temporal Crowd Behavior Analysis and Predictive Risk Management System**.

CrowdSentry ingests video from crowd-monitoring cameras, detects and tracks people, builds
spatio-temporal density and flow features, and predicts crowd-risk levels so operators can act
before a situation becomes dangerous.

> **Status: functional system; model validation pending.** The repository contains the
> end-to-end ingestion, feature extraction, risk scoring, persistence, API, and
> dashboard paths, and they run against real camera feeds. The risk model is trained
> on simulation data, so read the limitations before treating scores as an
> operational safety system.

---

## Project structure

```
CrowdSentry/
├── backend/                 # Python 3.12 + FastAPI service
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py    # env-driven settings
│   │   │   └── database.py  # engine, SessionLocal, declarative Base
│   │   ├── vision/
│   │   │   ├── detection.py # YOLOv8 person detection
│   │   │   ├── tracking.py  # ByteTrack IDs + velocity
│   │   │   ├── features.py  # windowed feature extraction
│   │   │   ├── source.py    # file / webcam / RTSP video abstraction
│   │   │   └── annotate.py  # debug drawing helpers
│   │   ├── ml/
│   │   │   ├── train_risk_model.py  # XGBoost training
│   │   │   └── predict.py           # predict_risk()
│   │   ├── routers/
│   │   │   ├── cameras.py   # list / detail / history
│   │   │   ├── alerts.py    # queue + acknowledge
│   │   │   ├── dashboard.py # header aggregates
│   │   │   └── live.py      # WS /ws/live
│   │   ├── services/
│   │   │   ├── realtime_processor.py  # per-camera background workers
│   │   │   └── broadcaster.py         # thread -> event-loop fan-out
│   │   ├── models.py        # document dataclasses (to_doc/from_doc)
│   │   ├── schemas.py       # Pydantic API contracts
│   │   └── main.py          # FastAPI app + /health endpoints
│   ├── scripts/
│   │   ├── seed.py          # sample cameras
│   │   ├── simulate_crowd_scenarios.py  # social-force training data
│   │   ├── test_detection.py# annotated detection run
│   │   └── test_tracking.py # annotated tracking run
│   ├── tests/
│   │   ├── test_features.py # synthetic-data unit tests
│   │   ├── test_risk_model.py   # Fruin labels + prediction contract
│   │   ├── test_realtime.py     # alerting, status, lifespan
│   │   ├── test_api.py          # schemas, routing, CORS, broadcaster
│   │   └── test_api_database.py # MongoDB-backed API integration tests
│   ├── ml_artifacts/        # trained model, metadata, charts
│   ├── data/
│   │   ├── samples/         # put your input videos here (git-ignored)
│   │   └── output/          # annotated results land here (git-ignored)
│   ├── .env.example
│   └── requirements.txt
├── frontend/                # React 19 + Vite dashboard
│   ├── src/
│   │   ├── api/client.js    # REST client + ws:// URL
│   │   ├── hooks/
│   │   │   ├── useLiveFeed.js   # websocket + reconnect
│   │   │   └── useDashboard.js  # all dashboard state
│   │   ├── lib/risk.js      # levels, colours, formatting
│   │   ├── components/
│   │   │   ├── Header.jsx       # counts + feed status
│   │   │   ├── AlertBanner.jsx  # open alerts + acknowledge
│   │   │   ├── CameraGrid.jsx   # camera cards
│   │   │   ├── CrowdMap.jsx     # SVG map by lat/long
│   │   │   ├── CameraDetail.jsx # recharts history (lazy)
│   │   │   └── RiskBadge.jsx
│   │   ├── test/App.test.jsx    # render smoke tests
│   │   ├── App.jsx
│   │   ├── index.css        # dark theme tokens
│   │   └── main.jsx
│   ├── .env.example
│   └── package.json
├── docker-compose.yml       # MongoDB 7 (local dev database)
└── .gitignore
```

### Why these pieces

| Layer | Choice | Reason |
| --- | --- | --- |
| API | FastAPI + Uvicorn | async, typed request/response models, auto-generated OpenAPI docs |
| Vision | OpenCV + Ultralytics YOLOv8 | person detection and multi-object tracking on video frames |
| Analytics | NumPy + SciPy | density fields, optical flow statistics, spatio-temporal aggregation |
| Prediction | scikit-learn + XGBoost | risk classification / regression over the engineered features |
| Storage | **MongoDB** | camera documents, time-series observations, GeoJSON points (2dsphere) — Atlas or a local mongod |
| UI | React + Vite | fast HMR dev loop for the operator dashboard |

---

## Prerequisites

- **Python 3.11+** (developed against 3.12)
- **Node.js 18+** (developed against 24)
- **Docker Desktop** (for the local MongoDB), or a MongoDB Atlas cluster you can reach via `MONGO_URI`

---

## Running the stack

Run the database first, then the backend and frontend in their own terminals. The
following is the complete first-run path from a fresh checkout.

### 1. Database (MongoDB)

```bash
docker compose up -d
```

This starts `crowdsentry-db` — a MongoDB 7 instance — on **localhost:27017**. To use
MongoDB Atlas instead, skip this step and set `MONGO_URI` in `backend/.env` to the
cluster's `mongodb+srv://` connection string.

Verify it is up:

```bash
docker compose exec db mongosh crowdsentry --quiet --eval "db.runCommand('ping')"
```

Useful commands:

```bash
docker compose logs -f db   # follow logs
docker compose down         # stop (data is kept in the mongodata volume)
docker compose down -v      # stop and DELETE all data
```

There is no migration step: the API creates the collections' indexes itself at
startup (`ensure_indexes` in `app/core/database.py`), idempotently.

### 2. Backend (FastAPI)

```bash
cd backend

# --- first time only ---
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt    # only needed to run the test suite
pip install -r requirements-train.txt  # only needed to train the risk model
cp .env.example .env          # Windows: copy .env.example .env

python -m scripts.seed        # insert 3 sample cameras

# Build model artifacts if ml_artifacts/risk_model.json is not present.
python -m scripts.simulate_crowd_scenarios
python -m app.ml.train_risk_model

# --- every time ---
uvicorn app.main:app --reload --port 8000
```

| URL | Purpose |
| --- | --- |
| http://localhost:8000/health | health check — `{"status":"ok",...}` |
| http://localhost:8000/docs | interactive Swagger UI |
| http://localhost:8000/redoc | ReDoc API reference |

> Note: `requirements.txt` pulls in `ultralytics`, which installs PyTorch — the first
> install downloads a few hundred MB and takes several minutes.

#### Historical baselines

After the processor has collected observations, compute the per-camera hourly/weekday
density baselines with:

```bash
python -m scripts.compute_baselines
```

The command is safe to run nightly because it upserts each `(camera, hour, weekday)`
slot. A simple UTC cron entry is:

```cron
15 3 * * * cd /path/to/CrowdSentry/backend && .venv/bin/python -m scripts.compute_baselines >> /var/log/crowdsentry-baselines.log 2>&1
```

Until a slot has a baseline (or its variance is zero), `historical_deviation` falls
back to `0.0`, so a new camera can be processed before its first baseline run.

### 3. Frontend (Vite + React)

```bash
cd frontend
npm install                   # first time only
cp .env.example .env          # optional; only if the API is not on localhost:8000
npm run dev
```

Open **http://localhost:5173**. The frontend reads `VITE_API_BASE_URL` from
`frontend/.env` when set; in development it otherwise calls the same-origin Vite
proxy, and a production build defaults to the same origin it is served from (see the
production deployment section — nginx proxies `/api` and `/ws` to the backend).

Open **http://localhost:5173** for the operator dashboard — see the Dashboard section below.

---

## Configuration

Both services read configuration from a local `.env` file; `.env.example` in each folder lists
every supported key. `.env` files are git-ignored.

**backend/.env**

| Key | Default | Description |
| --- | --- | --- |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string — set to the Atlas `mongodb+srv://` URI in production |
| `MONGO_DB` | `crowdsentry` | database name |
| `CORS_ORIGINS` | `http://localhost:5173,...` | comma-separated allowed origins |
| `API_KEY` | *(empty — auth off)* | shared API key; set before exposing the API to a network |
| `API_DOCS_ENABLED` | `false` | serve `/docs`, `/redoc` and `/openapi.json` |
| `TRUSTED_HOSTS` | *(empty — off)* | comma-separated allowed `Host` headers |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | uvicorn listen address (read by the launch command, not the app) |

Every `Settings` field in `app/core/config.py` is also settable as an environment
variable — `DETECTION_MODEL`, `REALTIME_ENABLED`, `FEATURES_WINDOW_SECONDS`, and so on.

**frontend/.env**

| Key | Default | Description |
| --- | --- | --- |
| `VITE_API_BASE_URL` | *(empty — same origin)* | API origin, when the dashboard is hosted separately |
| `VITE_API_KEY` | *(empty)* | baked into the bundle; must match the backend `API_KEY` |

---

## Production deployment

`docker-compose.prod.yml` builds and wires the API/worker container and an
nginx-served dashboard that reverse-proxies `/api` and `/ws`. The database is your
MongoDB deployment (Atlas or self-hosted) reached over `MONGO_URI` — no DB service
runs inside the compose file:

```bash
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/crowdsentry \
API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32)") \
docker compose -f docker-compose.prod.yml up -d --build
```

The dashboard is then on `http://localhost` (`WEB_PORT` to change it) and the API is
reachable only through the web service's proxy — `api` publishes no port to the host.

Before going further than a private network, work through this list:

- **Set `API_KEY`.** With it unset the API is completely unauthenticated — anyone who
  can reach it can read camera locations and acknowledge alerts. The same value is
  baked into the dashboard build (`VITE_API_KEY` build arg, fed from `API_KEY` in
  compose), so the browser can call the API. This is shared-secret auth, adequate for
  an ops dashboard on a controlled network; for per-user identity put an
  authenticating proxy (oauth2-proxy, an IdP-aware ingress) in front.
- **TLS terminates in front of nginx.** Put the `web` service behind your TLS
  terminator (a load balancer, Caddy, another nginx with certs). Browsers then
  upgrade `wss://` automatically — the frontend derives the socket scheme from the
  page origin.
- **Exactly one realtime-enabled API process.** The camera workers run inside the API
  process, so `uvicorn --workers 2` or two replicas with `REALTIME_ENABLED=true` would
  duplicate every feed's detection and double-write observations. To scale reads, run
  extra replicas with `REALTIME_ENABLED=false` behind the proxy and keep one
  realtime-enabled instance.
- **Indexes are created at startup.** `ensure_indexes()` runs in the app lifespan —
  `createIndex` is idempotent, so every replica can safely run it (there is no schema
  migration step with MongoDB).
- **`stream_url` is never served.** Camera objects expose only `stream_configured`
  (bool), and processor status logs/snapshots show the redacted URL — feed
  credentials embedded in `rtsp://user:pass@...` cannot leak through the API.
- **Docs stay dark.** `/docs` and `/openapi.json` are off unless
  `API_DOCS_ENABLED=true`.
- **Baselines need a scheduler.** Either set `BASELINE_AUTO_ENABLED=true` (runs inside
  the single API instance) or schedule `python -m scripts.compute_baselines`
  externally (cron/Task Scheduler) — the compose file deliberately does not enable
  the in-process job so scaling out never double-fires it.
- **Probes.** `/health` (liveness) and `/health/ready` (DB round-trip) are
  unauthenticated and safe for load balancers and orchestrators.

---

## Database schema

MongoDB collections; the document shape is defined by the dataclasses in
`backend/app/models.py` (`to_doc`/`from_doc`). Every document carries a public
integer `id` allocated from the `counters` collection, so API ids stay numeric.

```
cameras --+--> crowd_observations   raw per-window measurements from the vision pipeline
          +--> historical_baselines what "normal" looks like per (hour, weekday)
          +--> risk_scores          model output, one doc per scored window
          +--> alerts               operator-facing events raised from a score
```

| Collection | Purpose | Key fields |
| --- | --- | --- |
| `cameras` | one doc per fixed camera | `id` (unique int), `name` (unique), `location_name`, `latitude`/`longitude`, `location` (GeoJSON Point), `area_sq_meters`, `pixels_per_meter`, `stream_url`, `is_active`, `created_at` |
| `crowd_observations` | one measurement window | `id`, `camera_id`, `timestamp`, `person_count`, `density`, `mean_flow_speed`, `flow_direction_variance`, `optical_flow_entropy`, `stop_ratio`, `density_rate_of_change`, `historical_deviation` |
| `historical_baselines` | typical density per slot | unique key `(camera_id, hour_of_day, day_of_week)`, `avg_density`, `stddev_density`, `sample_count` |
| `risk_scores` | model output | `id`, `camera_id`, `timestamp`, `risk_score` (0-100), `risk_level`, `contributing_features` (subdocument) |
| `alerts` | operator-facing events | `id`, `camera_id`, `timestamp`, `risk_score`, `risk_level`, `message`, `acknowledged` |

**Density** is `person_count / cameras.area_sq_meters` (people/m²). **Deviation from normal**
joins an observation to its baseline slot — as an aggregation pipeline it is a `$lookup`
on `(camera_id, hour_of_day, day_of_week)`; in `mongosh`:

```js
db.crowd_observations.aggregate([
  {$match: {camera_id: 2}},
  {$lookup: {from: "historical_baselines",
    let: {cid: "$camera_id", ts: "$timestamp"},
    pipeline: [{$match: {$expr: {$and: [
      {$eq: ["$camera_id", "$$cid"]},
      {$eq: ["$hour_of_day", {$hour: "$$ts"}]},
      {$eq: ["$day_of_week", {$subtract: [{$isoDayOfWeek: "$$ts"}, 1]}]},
    ]}}}],
    as: "baseline"}},
  {$sort: {timestamp: -1}},
]);
```

### Indexes

Every child collection carries a compound `{camera_id, timestamp}` index, because the
dominant query is always "the recent window for one camera". `ensure_indexes()` in
`app/core/database.py` creates all of them idempotently at API startup:

| Index | Collection | Purpose |
| --- | --- | --- |
| `{camera_id, timestamp}` | `crowd_observations` | per-camera time series (the hot path) |
| `{timestamp}` | `crowd_observations` | cross-camera "what happened at time T" |
| `{camera_id, timestamp}` | `risk_scores` | per-camera risk history |
| `{risk_level, timestamp}` | `risk_scores` | recent HIGH/CRITICAL across all cameras |
| `{camera_id, timestamp}` | `alerts` | per-camera alert history |
| `{acknowledged, timestamp}` | `alerts` | the open-alert queue |
| `{id}` unique, `{name}` unique | `cameras` | public id and name lookups |
| `{location}` 2dsphere | `cameras` | GeoJSON proximity queries |
| `{camera_id, hour_of_day, day_of_week}` unique | `historical_baselines` | the slot-lookup key, upsert-safe |

### Spatial queries

`cameras.location` is a GeoJSON `Point` written alongside flat `latitude`/`longitude`
by `Camera.to_doc()`, so the two can never drift apart. A "cameras within 500 m of
CAM-03" query is a `$near` on the 2dsphere index:

```js
db.cameras.find({
  location: {$near: {$geometry: {type: "Point", coordinates: [77.6032, 12.9761]},
                     $maxDistance: 500}}
});
```

---

## Person detection

Per-frame YOLOv8 detection filtered to the COCO `person` class. No tracking yet: each
frame is independent, so a person in frame N has no identity linking them to frame N+1.

```python
from app.vision.detection import detect_people

boxes = detect_people(frame)          # frame is a BGR numpy array
print(len(boxes), "people")
print(boxes[0].confidence, boxes[0].foot_point)
```

`detect_people` shares one lazily-loaded model process-wide. Construct `PersonDetector`
directly for per-camera settings. Each `BoundingBox` carries `x1, y1, x2, y2, confidence`
plus `width`, `height`, `area`, `center`, `foot_point`, and `to_xywh()` — `foot_point`
(bottom-centre) is the one to homography-project onto a floor plan later, since the box
centre floats around chest height.

### Video sources

`app/vision/source.py` wraps the three input kinds behind one interface, so calling code
never branches on which it got:

```python
from app.vision.source import open_source

with open_source("data/samples/crowd.mp4") as src:   # file
with open_source(0) as src:                          # webcam
with open_source("rtsp://user:pass@cam-01/stream") as src:   # network camera

    for frame_index, frame in src.frames(stride=5):
        ...
```

`open_source` dispatches on the argument: a digit is a webcam index, an `rtsp://`/`http://`
URL is a stream, anything else is a file path. Live sources set `is_live = True`, keep an
OpenCV buffer of 1 so you process the newest frame rather than a queued stale one, and
reconnect automatically on a dropped stream. `stride=N` processes every Nth frame — the
cheapest way to cut cost, since crowd density barely changes between adjacent frames at
25 fps.

### Running the test script

Put a video with people in it at `backend/data/samples/` (any name; `.mp4` and `.avi` both
work — the folder is git-ignored, so sample footage never gets committed):

```bash
cd backend
python -m scripts.test_detection --source data/samples/crowd.mp4
```

The annotated video is written to `backend/data/output/<name>_detected.mp4` with a box and
confidence score on each person and a per-frame count in the corner. The run prints a
summary: frames processed, total/average/peak detections, and ms per frame.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--conf 0.25` | lower the confidence threshold (helps with distant/occluded people) |
| `--max-dimension 960` | inference resolution — the main speed/accuracy knob |
| `--max-frames 300` | stop early; **required** to terminate a webcam or RTSP source |
| `--stride 5` | process every 5th frame |
| `--model yolov8s.pt` | larger model: better recall on small people, slower |
| `--device cuda` | run on GPU |
| `--output path.mp4` | write somewhere other than the default |

```bash
# webcam, 200 frames
python -m scripts.test_detection --source 0 --max-frames 200

# network camera, lower threshold, every 3rd frame
python -m scripts.test_detection --source rtsp://user:pass@cam-01/stream     --conf 0.25 --stride 3 --max-frames 500
```

### Tuning notes

Defaults live in `app/core/config.py` and are all overridable from `backend/.env`:

| Setting | Default | Why |
| --- | --- | --- |
| `DETECTION_MODEL` | `yolov8n.pt` | smallest/fastest; downloaded and cached on first run (~6 MB) |
| `DETECTION_CONFIDENCE` | `0.35` | below the usual 0.5 on purpose — in dense crowds the occluded people at the back score low, and dropping them undercounts density exactly where it matters |
| `DETECTION_MAX_DIMENSION` | `640` | frames are downscaled so the longer side is at most this, and the network runs at this size |
| `DETECTION_IOU` | `0.55` | above the 0.45 default: people in a crowd genuinely overlap, and aggressive NMS deletes real detections |
| `DETECTION_DEVICE` | *(auto)* | set to `cuda` on a GPU machine |
| `TRACKING_TRACKER` | `bytetrack.yaml` | or `botsort.yaml` for appearance re-ID |
| `TRACKING_PIXELS_PER_METER` | `0` *(uncalibrated)* | fallback when a camera row has none |
| `TRACKING_VELOCITY_WINDOW_SECONDS` | `0.5` | history window for velocity, smooths box jitter |
| `FEATURES_WINDOW_SECONDS` | `5.0` | feature aggregation window |
| `FEATURES_STATIONARY_SPEED_M_S` | `0.2` | below this a person counts as stopped |
| `FEATURES_OPTICAL_FLOW_ENABLED` | `true` | Farneback flow entropy; the costliest feature |
| `FEATURES_OPTICAL_FLOW_BINS` | `16` | direction histogram bins (max entropy 4 bits) |

`max_dimension` is the cost knob. Measured on this machine (CPU-only PyTorch, yolov8n,
1440p input):

| `max_dimension` | ms/frame | notes |
| --- | --- | --- |
| 320 | 40 | ~2.5x faster, but starts missing people |
| 640 | 68 | default |
| 960 | 113 | |
| 1280 | 174 | diminishing returns unless people are very small in frame |

Two things to know before you scale this up:

- **This machine has CPU-only PyTorch** (`torch 2.14.0+cpu`), giving ~17 fps at 640px on a
  640x360 video. That is fine for offline analysis of recorded footage but below real time
  for multiple live cameras. Installing a CUDA build of PyTorch and setting
  `DETECTION_DEVICE=cuda` is the single biggest speedup available.
- **`yolov8n` is a general-purpose COCO model.** It is good at nearby, unoccluded people
  and weak on the small, heavily-overlapped ones that make up a genuinely dense crowd —
  which is exactly the regime this project cares about. Expect to move to `yolov8s`/`m`, and
  for very dense scenes, to compare against a crowd-counting/density-estimation approach
  rather than box detection.

---

## Person tracking

`app/vision/tracking.py` puts persistent IDs on the detections using Ultralytics'
built-in **ByteTrack**. That was the cleanest integration available: `model.track(persist=True)`
runs detection and association in a single call, so there is no second copy of the boxes
to keep in sync. ByteTrack also suits crowds specifically -- its two-stage association
matches high-confidence boxes first, then rescues *low*-confidence ones against the
leftover tracks, which is exactly the behaviour you want when people are half-occluded and
drop below threshold for a few frames without actually leaving.

```python
from app.vision.tracking import track_people

for tracked in track_people("data/samples/crowd.mp4", pixels_per_meter=52.0):
    print(tracked.timestamp, tracked.person_count)
    for person in tracked.people:
        print(person.track_id, person.centroid, person.speed_m_per_s)
```

`TrackedFrame` carries `frame_index`, `timestamp`, `source_time`, `people`, and the frame
itself. Each `TrackedPerson` has `track_id`, `bbox`, `centroid`, `foot_point`,
`velocity_px_per_s`, `speed_px_per_s`, `speed_m_per_s`, `age`, and `direction_radians`.

One tracker follows one stream. ByteTrack keeps its state inside the Ultralytics model
object, so two concurrent cameras need two `PersonTracker` instances -- share one and the
identities contaminate each other. `track_people()` builds a fresh tracker per call, which
is why it is the safe default.

### Velocity and calibration

Velocity is measured **from the foot point** (bottom-centre of the box), not the centroid:
the foot point is where the person meets the ground, so it is the only one that maps to a
real-world position. The box centre drifts up and down with posture and occlusion.
`centroid` is still exposed, it just is not what velocity uses.

It is also measured **across a window** (`tracking_velocity_window_seconds`, default 0.5s)
rather than between adjacent frames. Frame-to-frame differencing is swamped by box jitter --
a box that wobbles two pixels at 25 fps implies a phantom 50 px/s -- and the window averages
that out while still tracking genuine changes of pace.

Pixel speed becomes m/s via `pixels_per_meter`, resolved in this order:

1. the `pixels_per_meter` argument to `PersonTracker`
2. `cameras.pixels_per_meter` for that camera in the database
3. `TRACKING_PIXELS_PER_METER` in `.env`

With none of those set, `speed_m_per_s` is `None` -- deliberately, rather than reporting a
fabricated number. Calibrate by measuring a known real-world distance in the camera's own
footage (a door width, a run of floor tiles) and dividing its pixel length by its length in
metres.

> **A single constant assumes a roughly fronto-parallel view.** Under perspective, a metre
> near the camera spans far more pixels than a metre at the back of the scene, so one
> constant is an approximation that degrades as the camera tilts. It is acceptable for a
> shallow-angle view and increasingly wrong for a high mounted one. The proper fix is a
> homography onto a floor plan, which is also what you will want for density maps -- worth
> doing before any speed number feeds the risk model.

### Running the tracking script

```bash
cd backend
python -m scripts.test_tracking --source data/samples/crowd.mp4

# pull calibration from a seeded camera row
python -m scripts.test_tracking --source data/samples/crowd.mp4 --camera CAM-02-CONCOURSE

# or state it directly
python -m scripts.test_tracking --source data/samples/crowd.mp4 --pixels-per-meter 52
```

Output goes to `backend/data/output/<name>_tracked.mp4`: each person gets a colour derived
from their ID (stable for the whole run), a motion trail, a velocity arrow showing where
they would be in half a second, and a label with ID and speed.

**What to look for when checking track stability:**

- A person **keeps one colour and one ID** while visible. A colour change mid-walk is an
  identity switch -- the main thing this script exists to expose.
- **ID numbers stay low.** Rapidly climbing IDs mean tracks are being dropped and recreated
  rather than maintained.
- **Trails are smooth.** Jagged trails mean the detector is jittering frame to frame.
- **Arrows point where people are walking** and lengthen with pace.

The summary reports unique IDs, longest/average track lifetime, and how many tracks lasted
two frames or fewer -- that last number is the clearest signal of an unstable tracker, since
a two-frame track is almost always a false start rather than a person. If unique IDs run
well above the peak person count, the script suggests remedies.

Flags beyond the detection ones: `--tracker botsort.yaml` (appearance re-ID; recovers IDs
better through long occlusions, noticeably slower), `--camera NAME`, `--pixels-per-meter`,
and `--trail-seconds`.

### Measured behaviour

On a synthetic clip with a person moving at a known 150 px/s, tracking recovered
**150.2 px/s mean (sigma 0.9), a 0.1% error**, correct heading, and a single ID across all
90 frames. The same numbers held at `--stride` 1, 2 and 3, confirming velocity uses media
time rather than a frame counter. On the sample crowd clip: 5 unique IDs for a peak of 4
people, longest track spanning all 100 frames, no fleeting tracks.

Throughput is ~7 fps on this CPU-only machine (tracking adds association cost on top of
detection). As with detection, a CUDA build of PyTorch is the fix for real-time use.

---

## Feature extraction

`app/vision/features.py` is the join between the vision pipeline and the risk model. It
consumes the `TrackedFrame` stream, aggregates it into fixed time windows (5s by default),
and emits one `FeatureVector` per window, optionally writing each to `crowd_observations`.

```
tracked frames  ->  [ 5s window ]  ->  FeatureVector  ->  crowd_observations
                                                      ->  risk model (later)
```

```python
from app.vision.features import FeatureExtractor
from app.vision.tracking import track_people

extractor = FeatureExtractor(camera_id=2, persist=True)
for features in extractor.extract(track_people("data/samples/crowd.mp4")):
    print(features.density, features.historical_deviation)
```

`area_sq_meters` and `pixels_per_meter` are read from the camera row when not passed
explicitly. Pass `area_sq_meters` directly to run without a database at all.

### The features

| Feature | How it is computed |
| --- | --- |
| `density` | mean people per frame / `camera.area_sq_meters` |
| `density_rate_of_change` | change in density since the previous window, per second (signed) |
| `mean_flow_speed` | mean per-track speed; m/s when calibrated, else px/s |
| `flow_direction_variance` | `1 - R`, where `R` is the mean resultant length of movement headings (0 = everyone moving together, 1 = headings cancel out) |
| `optical_flow_entropy` | Shannon entropy (bits) of the Farneback dense-flow direction histogram, magnitude-weighted |
| `stop_ratio` | fraction of tracks in the window that were near-stationary |
| `historical_deviation` | `(density - avg_density) / stddev_density` from `historical_baselines` for the window's hour and weekday |

Four decisions in there are worth knowing, because each one is a place the obvious
implementation is wrong:

**Aggregation is per track, not per detection.** A person visible for all 50 frames of a
window would otherwise count 50 times toward the averages and drown out someone who walked
through in 5. Each track is reduced to one value first, then those are combined, so every
person counts once regardless of screen time.

**`person_count` is the per-frame mean, not the number of distinct track IDs.** Over 5
seconds people walk through, so distinct IDs measures *throughput*, not occupancy — and
density is an occupancy measure. Both are exposed (`person_count` and
`unique_track_count`); the gap between them is itself informative, since
`unique_track_count >> person_count` means a flowing crowd rather than a static one.

**Direction variance uses vectors, never angles.** Averaging angles directly is wrong at
the wrap-around: the mean of 359 degrees and 1 degree is 0, not 180. Stationary people are
excluded entirely — their heading is jitter, not intent, and including them would dilute a
genuine counter-flow signal.

**Optical flow entropy is magnitude-weighted and heavily downscaled.** Direction entropy is
a scene-level statistic, not a per-person one, so the flow field runs at 320px and samples
at most 4 frame pairs per window. Farneback is by far the most expensive feature here;
disable it with `compute_optical_flow=False` when frames are unavailable.

### Two things to be aware of

**`stop_ratio` was ambiguous in spec, so both readings are computed.** "Fraction whose speed
dropped below a threshold this window vs last window" can mean the *level* (how many are
stopped now) or the *transition* (how many just stopped). `stop_ratio` is the level — it
matches the column and is the more robust signal. `newly_stopped_ratio` is the transition:
of the tracks present in both windows, the fraction that were moving and are now stopped.
That one is the better leading indicator of a crowd seizing up, and it is `None` when no
tracks carried over. Only `stop_ratio` is persisted; both are on the `FeatureVector`.

**Uncalibrated cameras leave `mean_flow_speed` NULL.** Storing px/s for some cameras and m/s
for others in one column would silently corrupt any model trained across cameras, so the
column takes m/s or nothing. The `FeatureVector` always carries both
`mean_flow_speed_px_s` and `mean_flow_speed_m_s`. Set `cameras.pixels_per_meter` to get
real units.

### Running the tests

The backend suite covers feature extraction, model serving, realtime behavior, API
contracts, and database-backed API integration. The unit tests need no video or database;
the MongoDB integration tests insert marked documents and clean them up at teardown,
skipping when Mongo is not reachable:

```bash
cd backend
python -m pytest
```

Expected values are derived by hand in each test rather than by re-running the
implementation, so a wrong implementation fails rather than agreeing with itself. Coverage
includes the angle-wraparound case, per-track vs per-detection weighting, threshold
boundaries, window splitting and trailing partial windows, empty scenes, and every
`None`-returning edge case (no baseline, zero-variance baseline, no calibration, nobody
moving).

---

## Risk model

A 4-class XGBoost classifier over the seven features, producing a 0-100 risk score, a
LOW/MODERATE/HIGH/CRITICAL level, and per-prediction feature attributions.

```python
from app.ml.predict import predict_risk

risk = predict_risk(feature_vector)     # a FeatureVector, or a plain dict
print(risk.risk_level, risk.risk_score, risk.confidence)
for contribution in risk.top_features:
    print(contribution.feature, contribution.direction, contribution.contribution)

doc = risk.to_doc(camera_id=2)          # persistable app.models.RiskScore document
```

### Building it

```bash
cd backend
python -m scripts.simulate_crowd_scenarios     # ~4 min, writes 2400 labelled windows
python -m app.ml.train_risk_model              # ~2 s, writes ml_artifacts/
```

To retrain after changing the simulator, feature definitions, or labels, regenerate the
dataset and rebuild both artifacts:

```bash
cd backend
python -m scripts.simulate_crowd_scenarios --seed 42
python -m app.ml.train_risk_model --data data/simulated/crowd_windows.csv \
  --artifacts ml_artifacts --seed 42 --rounds 400
```

`train_risk_model` also accepts `--max-depth`, `--learning-rate`, and a different
`--artifacts` directory for experiments. Restart the backend after replacing the model
so its process-wide cached booster is reloaded. Always review the held-out metrics and
run `pytest` before using a newly trained artifact.

Artifacts land in `backend/ml_artifacts/`: `risk_model.json` (the booster),
`risk_model_metadata.json` (feature order, classes, metrics), `feature_importance.png`,
and `confusion_matrix.png`.

### Training data: a simulation, not real incidents

**Read this before trusting any number below.** There is no incident-labelled crowd
footage here, so the model is trained on a 2D social-force pedestrian simulation
(Helbing & Molnar 1995) labelled against Fruin's density standard. It has learned *what a
dangerous density and flow pattern looks like*, not *what preceded a real crush*. Before
this goes anywhere near an operational deployment it needs validation on real footage.

`scripts/simulate_crowd_scenarios.py` runs 400 scenarios through a corridor with a gate,
across four flow patterns — `unidirectional`, `bidirectional` (counter-flow), `converging`,
and `milling` — at occupancies spanning every Fruin band, and emits 2400 five-second
windows.

Crucially, features are computed by running the simulated agents through the **real
`FeatureExtractor`** — the same code that runs in production — so training features cannot
silently drift from serving features.

### Labels: Fruin LOS + a chaos escalation

Ground truth comes from Fruin's Level of Service for walkways (*Pedestrian Planning and
Design*, 1971), which defines bands by area per person in ft². Inverted to densities:

| LOS | ft²/person | ped/m² | Fruin's description | Our tier |
| --- | --- | --- | --- | --- |
| A | ≥ 35 | ≤ 0.31 | free flow, no conflicts | LOW |
| B | 25–35 | 0.31–0.43 | normal speeds, minor conflicts | LOW |
| C | 15–25 | 0.43–0.72 | speeds restricted, passing difficult | MODERATE |
| D | 10–15 | 0.72–1.08 | reverse flow causes serious conflict | MODERATE |
| E | 5–10 | 1.08–2.15 | shuffling, forward movement only | HIGH |
| F | < 5 | > 2.15 | contact unavoidable, movement breaks down | CRITICAL |

Fruin's standard is density-only, but a crowd at *moderate* density that has stopped moving
and lost a coherent direction is a documented crush precursor that density alone misses. So
a **chaos score** (`0.55 × flow_direction_variance + 0.45 × stop_ratio`) above 0.45 pushes
the label up one tier, capped at CRITICAL, and only above LOS B density — a lone person
wandering an empty plaza is not a risk however erratic. This fires on **13.5%** of windows.

### Results

Split is **grouped by scenario**, not by window: windows from one run are consecutive
slices of the same crowd, and splitting them randomly would put near-duplicates in both
train and test and report an accuracy that does not survive a new scene.

| Split | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| train (1440) | 0.988 | 0.986 | 0.988 |
| validation (480) | 0.971 | 0.966 | 0.971 |
| **test (480)** | **0.965** | **0.953** | **0.964** |

Per class (validation):

| Level | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| LOW | 0.988 | 0.994 | 0.991 | 172 |
| MODERATE | 0.936 | 0.970 | 0.953 | 135 |
| HIGH | 0.957 | 0.892 | 0.923 | 74 |
| CRITICAL | 1.000 | 0.990 | 0.995 | 99 |

Every misclassification on the test set is between *adjacent* tiers — the model never
confuses LOW with CRITICAL.

Feature importance (gain): `density` 54.6%, `optical_flow_entropy` 15.7%,
`historical_deviation` 10.3%, `flow_direction_variance` 7.9%, `mean_flow_speed` 5.8%,
`stop_ratio` 4.3%, `density_rate_of_change` 1.5%.

> **96.5% accuracy is not as impressive as it looks.** The labels are a deterministic
> function of density and chaos, and both are model inputs — so the model is largely
> recovering the labelling rule from its own features. It measures "can XGBoost learn a
> rule from its inputs", not "can it predict real crowd risk". The honest reading is that
> the pipeline is sound end to end; the accuracy number is not evidence about reality.

### Score and level

The 0-100 score is the probability-weighted mean of per-tier anchors (12.5 / 37.5 / 62.5 /
87.5) rather than a separate regressor. Score and level therefore can never contradict each
other, and the score inherits the model's uncertainty: a window the model is torn between
HIGH and CRITICAL scores between the two, which is what an operator dashboard wants.

### Explainability

`top_features` uses XGBoost's `pred_contribs`, which computes exact SHAP values for tree
ensembles with no extra dependency. These are *per-prediction* attributions — how much each
feature moved *this* window's score — not global importance, which is identical for every
window and useless as an operator explanation. `contributing_features()` returns a
JSON-ready blob for the `risk_scores.contributing_features` column.

### Known weaknesses

- **`optical_flow_entropy` is the weakest link.** In production it is the entropy of a
  dense Farneback flow-direction histogram; in training there are no images, so it is the
  entropy of the *agents'* heading histogram. Same definition, same units (bits over 16
  bins, magnitude-weighted), different source. It is the model's second most important
  feature at 15.7% gain, so this train/serve gap matters and should be measured against
  real footage before the feature is trusted.
- **Simulated dynamics are not real dynamics.** The simulation reproduces the empirical
  speed-density relationship, but it has no groups, no luggage, no panic, no terrain.
- **`mean_flow_speed` needs calibration.** Uncalibrated cameras feed the model NaN for it.

---

## Realtime processing service

`app/services/realtime_processor.py` runs the whole pipeline continuously, one worker per
active camera:

```
VideoSource -> PersonTracker -> FeatureExtractor -> predict_risk
                                     |                   |
                              crowd_observations     risk_scores
                                                         |
                                                alerts (HIGH/CRITICAL)
```

Workers start with the API and stop with it, via FastAPI's `lifespan`. A camera gets a
worker when `is_active` is true **and** `stream_url` is set; `stream_url` takes anything
`open_source` understands — an RTSP URL, a webcam index, or a path to a video file.

```bash
cd backend
uvicorn app.main:app --reload      # starts the API and the camera workers
curl http://127.0.0.1:8000/api/processor/status
```

`REALTIME_ENABLED=false` runs a pure API process with no workers.

### Threading, and why

The pipeline is entirely blocking CPU work — YOLO inference, OpenCV decoding, synchronous
PyMongo writes. Each camera therefore runs its **whole pipeline in its own thread** via
`asyncio.to_thread`, and the async layer only supervises: start, stop, restart-on-failure,
report status. torch, OpenCV and PyMongo all release the GIL during their heavy work, so
the threads genuinely run in parallel.

Measured with two cameras processing video, `/health` returns in **3.2 ms median, 5.0 ms
p95** — the event loop stays free.

> If you benchmark this yourself on Windows, use `127.0.0.1` rather than `localhost`.
> `localhost` resolves to IPv6 first and adds a flat ~2 s per request *in the client*,
> which looks exactly like a blocked server. It cost me a wrong diagnosis.

### Alerting

A window at HIGH or CRITICAL creates an `alerts` row:

```
High crowd risk detected near Demo Stadium North Gate - risk 66/100,
density 1.58 people/m2, driven by flow_direction_variance
```

Raw level-crossing would emit an alert every window — 12 a minute for a camera parked at
HIGH — so alerts are de-duplicated by three rules:

- **Cooldown.** At most one alert per `REALTIME_ALERT_COOLDOWN_SECONDS` (default 120).
- **Escalation always alerts.** HIGH → CRITICAL is new information and is never suppressed.
- **Sustained calm re-arms.** After `REALTIME_ALERT_REARM_WINDOWS` (default 2) consecutive
  non-alerting windows the cooldown is cleared entirely, so a crowd that calms down and
  later surges again alerts immediately. Requiring *two* windows debounces a camera
  flapping across the HIGH boundary, which would otherwise alert every other window.

### Structured logging

Every line carries `key=value` context, greppable without a log pipeline:

```
18:21:32 INFO  realtime  window scored   camera=CAM-01 level=CRITICAL score=65.73
                                         confidence=0.47 density=1.576 people=3.9
                                         stop_ratio=0.75 driver=flow_direction_variance ms=1354
18:21:32 WARN  realtime  ALERT raised    camera=CAM-01 level=CRITICAL score=65.73
```

### Resilience

A camera whose source fails is retried after `REALTIME_RESTART_DELAY_SECONDS`, up to
`REALTIME_MAX_CONSECUTIVE_FAILURES` times, then marked `failed` and left alone. A
successful open resets the counter, so intermittent dropouts over hours do not accumulate.
One camera failing never affects the others. A missing risk model stops that camera
immediately rather than retrying — no amount of reconnecting will produce a model.

| Setting | Default | Purpose |
| --- | --- | --- |
| `REALTIME_ENABLED` | `true` | start workers with the API |
| `REALTIME_FRAME_STRIDE` | `3` | process every Nth frame |
| `REALTIME_ALERT_COOLDOWN_SECONDS` | `120` | minimum gap between alerts per camera |
| `REALTIME_ALERT_REARM_WINDOWS` | `2` | calm windows before the cooldown clears |
| `REALTIME_RESTART_DELAY_SECONDS` | `5` | wait before retrying a failed source |
| `REALTIME_MAX_CONSECUTIVE_FAILURES` | `5` | give up on a camera after this many |
| `REALTIME_TORCH_THREADS` | `2` | cap intra-op threads so cameras do not oversubscribe |
| `LOG_LEVEL` | `INFO` | console log level |

---

## API

All endpoints are under `/api`, with interactive docs at `/docs`. CORS is configured for
the Vite dev server on both `localhost` and `127.0.0.1` (they are distinct origins to a
browser, and missing one breaks the dev server in confusing ways).

### REST

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cameras` | every camera with its latest risk score (`?active_only=true` to filter) |
| GET | `/api/cameras/{id}` | camera detail: latest risk, latest measurement, open alert count |
| GET | `/api/cameras/{id}/history?from=&to=&limit=` | observation and risk-score series for charting |
| GET | `/api/alerts?acknowledged=false` | the open alert queue, newest first |
| POST | `/api/alerts/{id}/acknowledge` | mark an alert acknowledged (idempotent) |
| GET | `/api/dashboard/summary` | header aggregates |
| GET | `/api/processor/status` | per-camera worker state |

```bash
curl http://127.0.0.1:8000/api/cameras
curl "http://127.0.0.1:8000/api/cameras/2/history?from=2026-09-03T00:00:00Z&to=2026-09-04T00:00:00Z"
curl "http://127.0.0.1:8000/api/alerts?acknowledged=false"
curl -X POST http://127.0.0.1:8000/api/alerts/9/acknowledge
```

Response shapes live in `app/schemas.py`, deliberately separate from the ORM models so the
database can change shape without silently changing what clients receive.

Four details worth knowing, because each is a place the obvious implementation misleads:

**`/api/cameras` is two aggregations + one find, not N+1.** The latest score and latest
observation per camera each come from a single `$sort`+`$group` aggregation over the
`(camera_id, timestamp)` index, merged into the camera list in memory — so listing 50
cameras costs three round trips rather than 51.

**History returns two parallel series, not a join.** A window can produce an observation
without a score (or the reverse), and joining them here would silently drop those rows.
When a range holds more rows than `limit`, the *newest* are kept and then flipped to
chronological order — truncating the recent end would defeat the purpose — and `truncated`
says so explicitly.

**`from` is a Python keyword**, so it is exposed via a query alias. `from`/`to` accept
ISO-8601; naive datetimes are treated as UTC rather than letting the database assume a
server timezone. The default range is the last hour.

**The dashboard summary only counts *recent* data.** A camera whose feed died an hour ago
still has a most-recent observation in the table; counting it would quietly inflate the
live headcount and hold the risk level at whatever it was when the feed stopped. Anything
older than `freshness_seconds` (default 120) is excluded and reported separately as
`stale_cameras`.

```json
{
  "generated_at": "2026-09-03T13:07:27Z",
  "total_cameras": 3, "active_cameras": 1,
  "reporting_cameras": 1, "stale_cameras": 0,
  "total_people": 3, "mean_density": 1.39, "peak_density": 1.39,
  "cameras_by_risk_level": {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 1},
  "highest_risk_level": "CRITICAL",
  "unacknowledged_alerts": 1,
  "freshness_seconds": 120.0
}
```

### WebSocket: `/ws/live`

Connect and receive a frame every time any camera produces a new risk score — no polling.

```js
const ws = new WebSocket("ws://localhost:8000/ws/live");
// With API_KEY configured on the backend, append it: /ws/live?api_key=<key>
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "risk_score") updateCamera(msg);
};
```

```
->  {"type": "hello", "server_time": "...", "subscribers": 1, "cameras": [...]}
->  {"type": "risk_score", "camera_id": 2, "camera_name": "CAM-02-CONCOURSE",
     "timestamp": "...", "risk_score": 71.69, "risk_level": "CRITICAL",
     "person_count": 3.21, "density": 1.2857, "top_feature": "density"}
->  {"type": "keepalive"}          every 25s when idle
->  {"type": "pong"}               in reply to a client "ping"
```

The `hello` frame carries the current camera list so a freshly-opened dashboard renders
immediately instead of showing an empty screen until the first window completes — which
with a 5 s window is visibly dead air.

**The threading matters here.** Risk scores are computed inside camera worker *threads*,
while WebSocket sends must happen on the event loop. `broadcaster.publish_threadsafe()`
bridges the two: it hands the event to the loop and returns immediately, so a slow or
stalled dashboard can never slow down the processing pipeline. Each subscriber gets its own
bounded queue, and when a client falls behind the *oldest* event is dropped rather than the
newest — for live data a stale reading is worthless and the current one is the whole point.
Publishing with no loop bound or nobody listening is a cheap no-op: processing must never
depend on a dashboard being connected.

> CORS does not apply to WebSockets — the handshake is exempt, so a browser can open
> `/ws/live` from any origin. The socket is therefore authenticated separately: when
> `API_KEY` is set, handshakes without `?api_key=<key>` (or an `X-API-Key` /
> `Authorization: Bearer` header) are refused with close code 4401.

### Verified

Against a live camera worker: all six REST endpoints, 404s for unknown camera and alert,
422 for an inverted date range, idempotent acknowledge, CORS preflight from the Vite
origin, and three concurrent WebSocket clients all receiving the same broadcasts.
`/api/dashboard/summary` measured 24 ms median with clients attached.

---

## Limitations & Future Work

- **Synthetic training data.** The current classifier is trained on simulated pedestrian
  scenarios with Fruin-derived labels, not real labeled incidents. A representative,
  privacy-preserving dataset of real crowd conditions and incidents would improve both
  accuracy and validation confidence.
- **Per-window prediction.** The model classifies each feature window independently. An
  LSTM-based temporal model (or another sequence model) is the next step for learning
  buildup, persistence, and precursors across multiple windows rather than relying on
  manually derived rate features alone.
- **Privacy.** CrowdSentry does not perform facial recognition or biometric
  identification. Track IDs are anonymous, short-lived identifiers used only to follow
  detections across frames; no facial embeddings or identity data are stored.

---

## Dashboard

A dark-themed monitoring view at `http://localhost:5173`, built on the REST API and
`/ws/live`.

```bash
cd backend && uvicorn app.main:app --reload    # terminal 1
cd frontend && npm run dev                     # terminal 2
```

| Region | What it shows |
| --- | --- |
| **Header** | people tracked, cameras reporting, peak density, a LOW/MODERATE/HIGH/CRITICAL camera breakdown, and live-feed status |
| **Alert banner** | unacknowledged HIGH/CRITICAL alerts, each with an Acknowledge button |
| **Crowd map** | camera markers positioned by latitude/longitude, coloured by risk, sized by headcount |
| **Camera grid** | one card per camera: name, live person count, density, and a coloured risk badge |
| **Detail view** | click any camera or marker for a recharts density + risk chart over 15m/1h/6h/24h |

### Live updates without polling

The dashboard loads `/api/dashboard/summary`, `/api/cameras` and `/api/alerts` once, then
opens `/ws/live`. Every `risk_score` frame **patches the matching camera in place** — no
refetch, so a busy venue costs one websocket frame per camera per window rather than a
round of HTTP requests.

The header counts are *derived from that same camera state* rather than re-read from the
server, so a card and the header can never disagree with each other.

Alerts are the exception: they are not pushed over the socket, so a HIGH/CRITICAL frame
triggers a debounced re-read of the alert queue (a burst across cameras costs one request,
not one each). A 30 s background reconcile covers anything missed while the socket was
down, and the socket itself reconnects with exponential backoff — a dashboard left open
overnight survives a backend restart without a refresh.

### Staleness is shown, not hidden

A camera whose feed stops still has a "latest" reading in the database. Presenting that as
current would quietly misreport the venue, so the UI applies the same freshness rule the
API does: anything older than `freshness_seconds` (default 120) is greyed out, labelled
*no data*, and excluded from the headline counts. A camera showing 99 people three hours
ago contributes 0 to "people tracked", not 99.

### Why SVG and not Leaflet

You asked whether Leaflet was worth adding — I don't think it is here. These cameras span a
few hundred metres, and at that scale street tiles carry no information the operator needs,
while adding a network dependency (and for most tile providers an API key) to a view that
should keep working in a control room with no internet. What the map actually needs is
relative position and risk colour, which is geometry rather than cartography. The projection
is ~30 lines in `CrowdMap.jsx`, corrects for longitude compression by latitude, and handles
the degenerate case of cameras sharing a position. If you later want a real floor plan or
satellite basemap, that component is the only thing that changes.

### Notes

- **Recharts is code-split.** It is ~390 kB and only the detail view needs it, so it loads
  on first open rather than in the initial bundle — 209 kB vs 598 kB to first paint.
- **Dark theme only.** Control rooms are dim, and the risk palette is tuned for contrast
  against a dark ground; a light theme would need a second palette rather than an inversion.
- **Only CRITICAL animates.** If every elevated state pulsed, none would stand out.
  `prefers-reduced-motion` disables the animation entirely.
- **Acknowledging is optimistic** — the alert leaves the queue immediately and is restored
  if the request fails.

### Frontend tests

```bash
cd frontend
npm test
```

Ten render tests mount the real component tree against payloads shaped exactly like the
API's, covering the live-update path (a websocket frame patches a card without refetching),
staleness, optimistic acknowledge, the detail view, an unreachable backend, no cameras, and
a camera that has never been scored. A companion script checked every field the components
read against the live API, so a renamed field fails loudly rather than rendering blank.

---

## Roadmap

1. ~~**Ingestion** — video source adapters (file, RTSP), frame sampling.~~ *(done)*
2. ~~**Detection & tracking** — YOLOv8 person detection with persistent track IDs.~~ *(done)*
3. ~~**Spatio-temporal features** — density, flow, dwell and turbulence features per time window.~~ *(done)*
4. ~~**Storage** — MongoDB collections, indexes ensured at startup.~~ *(done)*
5. ~~**Risk model** — XGBoost classifier over feature windows.~~ *(done, on simulated data)*
6. ~~**Dashboard** — live map, per-camera risk, history charts and threshold alerts.~~ *(done)*
7. **Validation on real footage** — the model is trained entirely on simulation; this is the gap that matters most before any real use.
