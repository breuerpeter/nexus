"""The one Rerun recording the whole sim logs to, as the Architecture page's "Observability and
recording" section describes.

`Logger` owns the recording, the gRPC server on :9876 or the ``.rrd``, writes the Recorder's
histories into it a block at a time, the series, the scene and the flown path, and routes the
``newton`` logger's events into the same recording. It exposes the shared log calls ``set_time``,
``log_image`` and the rest; each component logs its live rows through the ``ScopedLogger`` the
orchestrator hands it, which puts the component's path before each row's name.
"""

from .rerun_logging import (
    APP_ID,
    GRPC_PORT,
    RECORDING_ID,
    SERVER_URI,
    Logger,
    ScopedLogger,
    attach_log_handler,
    build_logger,
    recording_path,
    scene_only_blueprint,
)

__all__ = [
    "APP_ID",
    "GRPC_PORT",
    "RECORDING_ID",
    "SERVER_URI",
    "Logger",
    "ScopedLogger",
    "attach_log_handler",
    "build_logger",
    "recording_path",
    "scene_only_blueprint",
]
