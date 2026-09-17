"""The container runner: that a peer's console reaches the log file the caller named."""

import time

import nexus._src.containers as containers


class _FakeContainer:
    def __init__(self, chunks):
        self._chunks = chunks

    def logs(self, **kwargs):
        yield from self._chunks


def test_run_container_pumps_logs_to_the_file(monkeypatch, tmp_path):
    """The peer's console is a *file* the caller can scan afterwards: the PX4 warnings gate reads it
    out of ``sim.artifacts()``, so a run that logged nothing would pass the gate vacuously.
    """
    container = _FakeContainer([b"INFO  [px4] booting\n", b"WARN  [mag] interference\n"])

    class FakeClient:
        class containers_:
            @staticmethod
            def run(image, **kwargs):
                return container

        containers = containers_

    monkeypatch.setattr(containers, "client", lambda: FakeClient())

    log = tmp_path / "nested" / "px4.log"  # the dir doesn't exist yet
    containers.run_container(image="example.invalid/px4", command=["true"], log_path=str(log))

    end = time.monotonic() + 5  # the pump runs on a daemon thread
    while time.monotonic() < end and not (log.exists() and log.read_bytes().count(b"\n") == 2):
        time.sleep(0.01)
    assert log.read_text() == "INFO  [px4] booting\nWARN  [mag] interference\n"
