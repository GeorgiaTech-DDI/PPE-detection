import time

class FrameRateTracker:
    """Tracks frame count and elapsed time to compute real-time FPS."""

    def __init__(self) -> None:
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        self._count = 0
        self._start_time = time.perf_counter()

    def increment(self, n: int = 1) -> None:
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        return (
            f"Processed {self.count} frames at {self.fps:.2f} FPS, "
            f"Total time: {self.elapsed:.2f} seconds"
        )