"""Application configuration, loaded from the environment (or a local .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "CrowdSentry API"
    version: str = "0.1.0"

    # --- API surface ------------------------------------------------------------
    # Shared-secret authentication. Empty means "no auth" -- the right default for
    # local development, and wrong for anything reachable over a network. Clients
    # send `X-API-Key: <key>` or `Authorization: Bearer <key>`; browsers use the
    # ?api_key= query parameter on the WebSocket handshake (they cannot set headers).
    api_key: str = ""

    # Interactive docs (/docs, /redoc, /openapi.json) enumerate the whole API. Off by
    # default so a deployed instance does not advertise its surface; enable in .env
    # for development.
    api_docs_enabled: bool = False

    # Comma-separated Host headers to serve (TrustedHostMiddleware). Empty disables
    # the check; set it in production to stop Host-header poisoning on absolute URLs.
    trusted_hosts: str = ""

    # MongoDB. Local dev points at the compose service; production points at Atlas --
    # set MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/crowdsentry in the
    # environment, never in a committed file.
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "crowdsentry"
    # Both spellings: Vite prints localhost, but a browser (or a curl test) may use
    # 127.0.0.1, and the two are distinct origins to CORS.
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173"
    )

    # --- Person detection -------------------------------------------------------
    # Ultralytics weights. A bare filename is downloaded and cached on first use;
    # yolov8n is the smallest/fastest, yolov8s/m trade speed for recall on small,
    # distant people.
    detection_model: str = "yolov8n.pt"

    # Minimum confidence for a detection to count. Deliberately below the usual 0.5:
    # in dense crowds the occluded people at the back score low, and dropping them
    # undercounts density exactly where it matters most.
    detection_confidence: float = 0.35

    # Frames are downscaled so the longer side is at most this many pixels before
    # inference. The main speed/accuracy knob; 640 matches YOLOv8's training size.
    detection_max_dimension: int = 640

    # NMS IoU threshold. Higher than the 0.45 default because people in a crowd
    # genuinely overlap, and aggressive suppression deletes real detections.
    detection_iou: float = 0.55

    # "cpu", "cuda", "0", ... Empty lets Ultralytics choose.
    detection_device: str = ""

    # --- Tracking ---------------------------------------------------------------
    # Ultralytics tracker config: "bytetrack.yaml" or "botsort.yaml". ByteTrack is
    # motion-only and fast; BoT-SORT adds appearance re-identification, which recovers
    # IDs better through long occlusions at noticeably higher cost.
    tracking_tracker: str = "bytetrack.yaml"

    # Ground-plane scale used to convert pixel velocity to m/s. 0 means uncalibrated,
    # and speeds are reported as None rather than as a fabricated number. This is a
    # per-camera property -- see cameras.pixels_per_meter; this is only the fallback.
    tracking_pixels_per_meter: float = 0.0

    # Velocity is measured across this much track history rather than between adjacent
    # frames, where box jitter would swamp the real motion.
    tracking_velocity_window_seconds: float = 0.5

    # --- Feature extraction -----------------------------------------------------
    # Length of the aggregation window. Long enough to average out detector noise,
    # short enough that a surge is caught while it still matters.
    features_window_seconds: float = 5.0

    # Below this speed a person counts as "near-stationary" for stop_ratio. Applied in
    # m/s when the camera is calibrated, otherwise in px/s.
    features_stationary_speed_m_s: float = 0.2
    features_stationary_speed_px_s: float = 10.0

    # Directions slower than this contribute no heading to the circular variance --
    # a stationary person's direction is jitter, not intent.
    features_moving_speed_m_s: float = 0.15
    features_moving_speed_px_s: float = 8.0

    # Dense optical flow (Farneback) settings. Entropy is computed over a direction
    # histogram of the flow field.
    features_optical_flow_enabled: bool = True
    features_optical_flow_bins: int = 16
    # Farneback cost scales with pixel count; the flow field is heavily downscaled
    # because direction entropy is a scene-level statistic, not a per-person one.
    features_optical_flow_max_dimension: int = 320
    # Flow vectors shorter than this are noise, not motion, and are excluded.
    features_optical_flow_min_magnitude: float = 0.5
    # At most this many frame pairs per window, evenly spaced. Farneback is the most
    # expensive feature here by a wide margin.
    features_optical_flow_max_pairs: int = 4

    # --- Historical baselines ---------------------------------------------------
    # How much history a baseline is built from. Crowd patterns drift with seasons,
    # timetables and events, so "normal" should reflect recent weeks rather than
    # everything ever recorded. Four weeks gives roughly four samples per weekday slot.
    baseline_lookback_days: int = 28

    # Slots with fewer observations than this are skipped. A slot built from a single
    # sample has a standard deviation of zero, which makes every later reading
    # infinitely deviant.
    baseline_min_samples: int = 5

    # Run the baseline job inside the API process once a day. Off by default: an
    # external scheduler (cron, Task Scheduler, a systemd timer) still runs when the API
    # is down, logs where your other jobs log, and does not fire once per replica when
    # the API is scaled out. This exists for single-node deployments where adding a cron
    # entry is more friction than it is worth.
    baseline_auto_enabled: bool = False
    baseline_auto_hour_utc: int = 3

    # How long a FeatureExtractor may reuse a baseline lookup. This bounds staleness in
    # both directions: a worker that started before the nightly job would otherwise
    # cache the *absence* of a baseline for its whole lifetime and keep reporting the
    # cold-start default long after real baselines existed.
    features_baseline_cache_seconds: float = 300.0

    # Value used for historical_deviation when a camera has no baseline for the current
    # slot yet -- the cold-start case, which lasts until the job has run over real
    # history. Zero means "no evidence of deviation", the neutral reading.
    features_historical_deviation_default: float = 0.0

    # --- Realtime processing ----------------------------------------------------
    log_level: str = "INFO"

    # Start camera workers when the API starts. Turn off for a pure-API process, or
    # when running the processor separately.
    realtime_enabled: bool = True

    # Process every Nth frame. Detection is the bottleneck: at ~60 ms/frame on CPU a
    # worker cannot keep up with 25 fps, and for a 5 s window it does not need to --
    # density barely changes between adjacent frames.
    realtime_frame_stride: int = 3

    # Restart a file-backed camera from the beginning when the clip ends. Off by default
    # because real cameras are live feeds and a file running out is meaningful; turn it
    # on to demo the stack from sample footage without the feed stopping.
    realtime_loop_video: bool = False

    # Minimum gap between alerts for one camera. Without it a camera sitting at HIGH
    # emits an alert every window (12/minute) and the operator queue becomes unreadable.
    # An escalation to a higher level always alerts, cooldown or not.
    realtime_alert_cooldown_seconds: float = 120.0

    # Consecutive non-alerting windows before the cooldown is fully re-armed. A crowd
    # that genuinely calms down and then surges again is a new event and must alert
    # immediately -- but re-arming on a single calm window would let a camera flapping
    # across the HIGH boundary alert every other window. Two windows debounces that.
    realtime_alert_rearm_windows: int = 2

    # Wait before restarting a worker whose source failed, and how many consecutive
    # failures to tolerate before giving up on that camera.
    realtime_restart_delay_seconds: float = 5.0
    realtime_max_consecutive_failures: int = 5

    # How long to wait for workers to finish on shutdown before abandoning them.
    realtime_shutdown_timeout_seconds: float = 20.0

    # Intra-op threads PyTorch may use per inference. Torch defaults to half the
    # machine's cores (6 of 12 here) *per process*, so with several cameras the box is
    # heavily oversubscribed: 4 cameras would ask for 24 threads on 12 cores. Each
    # camera worker is already its own thread, so parallelism across cameras matters
    # more than parallelism within one inference. 0 leaves torch's default alone.
    #
    # This is a precaution against oversubscription, not a fix for a measured stall:
    # with one camera running, /health measures ~1 ms median either way.
    realtime_torch_threads: int = 2

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def trusted_host_list(self) -> list[str]:
        return [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]


settings = Settings()
