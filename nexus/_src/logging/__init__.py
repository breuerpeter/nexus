"""newton-logging: the one Rerun recording the whole sim logs to, as architecture.md §10 describes.

`Logger` owns NVIDIA Newton's ``ViewerRerun``, which is the in-process scene logging, the
gRPC server on :9876 and the ``.rrd``, and routes the ``newton`` logger's events into
the same recording. It exposes the shared log calls ``log_state``, ``set_time`` and ``log_image``;
each component logs its own quantities via its log step.
"""

from .rerun_logging import (
    APP_ID,
    FPV_ENTITY,
    GRPC_PORT,
    RECORDING_ID,
    SERVER_URI,
    Logger,
    attach_log_handler,
    build_logger,
    recording_path,
    scene_only_blueprint,
)

__all__ = [
    "APP_ID",
    "FPV_ENTITY",
    "GRPC_PORT",
    "RECORDING_ID",
    "SERVER_URI",
    "Logger",
    "attach_log_handler",
    "build_logger",
    "recording_path",
    "scene_only_blueprint",
]
