"""The scene-handler contract. See the package docstring for when a scene needs one."""

from __future__ import annotations


class SceneHandler:
    """Lifecycle hooks a scene type can override; every hook is an optional no-op.

    The hooks are render-frame hooks: on a runtime without a renderer the handler simply never
    runs. Scenes stay runtime-agnostic: the same scene flies anywhere, visibly where rendered.

    ``matches(usd_path)`` claims a scene by the **content** of its Universal Scene Description (USD)
    file. When a handler claims the scene, the physics stage compose skips the generic scene
    reference, and the handler owns composition:

    * :meth:`compose`: run the scene's live setup at render-frame construction. The scene USD is
      already open as the root stage; runtime injections belong on the session layer.
    * :meth:`follow`: per rendered frame, with the current body poses, for example a
      Level Of Detail (LOD) camera follow.
    * :meth:`on_ready`: the pre-lockstep quiet window, after the render warm-up and drain.
    * :attr:`keeps_default_viewport`: ``True`` if the handler reuses the boot viewport, which the
      frame then must not freeze.
    """

    keeps_default_viewport = False

    @staticmethod
    def matches(usd_path: str) -> bool:
        return False

    def compose(self, frame, stage) -> None:
        """Compose the scene's world onto ``stage``; ``frame`` is the owning ``RtxFrame``."""

    def follow(self, frame, body_q) -> None:
        """Per-render hook with the current body poses, a host copy."""

    def on_ready(self, frame) -> None:
        """Pre-lockstep hook, after the frame's warm-up and drain, for example streaming waits or probes."""
