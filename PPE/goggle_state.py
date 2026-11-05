from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Observations older than this are dropped from a track's window. Time based rather
# than frame based, since the frame rate drops as more faces need goggle inference.
WINDOW_S = 1.5
# A track needs this many observations in its window before a decision is made.
MIN_OBSERVATIONS = 5
# Hysteresis thresholds on the fraction of frames in the window with safety glasses.
# Becoming WEARING is harder than losing it.
ON_THRESHOLD = 0.7
OFF_THRESHOLD = 0.4


class WearState(Enum):
    UNKNOWN = "UNKNOWN"
    WEARING = "WEARING"
    NOT_WEARING = "NOT_WEARING"


@dataclass
class GoggleState:
    """
    Represents the state of a subject for which we are detecting goggles.
    track_id: ByteTrack ID for the subject
    observations: (timestamp, score) pairs within the last WINDOW_S seconds
    state: Current decision for the subject
    frames_since_last_face: Amount of frames input since last time this subject was seen
    """
    track_id: int
    observations: deque[tuple[float, float]] = field(default_factory=deque)
    state: WearState = WearState.UNKNOWN
    frames_since_last_face: int = 0

    @property
    def window_score(self) -> Optional[float]:
        """Mean score over the window, or None if there are no observations."""
        if not self.observations:
            return None
        return sum(score for _, score in self.observations) / len(self.observations)


def score_frame(detections: dict, safety_glasses_class: int) -> float:
    """
    Scores a single frame's goggle detections for one face.

    Params:
        detections: extract_detections output for the face crop.
        safety_glasses_class: Class index of the safety glasses label.
    Returns:
        1.0 if the highest scoring detection is safety glasses, otherwise 0.0. Regular
        eyeglasses and frames where nothing was found both count as 0.0.
    """
    # extract_detections sorts detections by score, highest first.
    if detections["num_detections"] == 0:
        return 0.0
    return 1.0 if detections["detection_classes"][0] == safety_glasses_class else 0.0


class GoggleStateTracker:
    """
    Aggregates per-frame goggle scores into a short term state for each track.

    update() must be called once for every frame that went through ByteTrack, including
    frames with no faces, so that eviction counts frames the same way ByteTrack does.
    """

    def __init__(self,
                 evict_after_frames: int,
                 window_s: float = WINDOW_S,
                 min_observations: int = MIN_OBSERVATIONS,
                 on_threshold: float = ON_THRESHOLD,
                 off_threshold: float = OFF_THRESHOLD):
        """
        Params:
            evict_after_frames: Tracks unseen for more than this many frames are removed.
                Should match ByteTrack's lost track buffer, after which the ID is gone.
            window_s: Length of each track's observation window in seconds.
            min_observations: Observations needed in the window before deciding.
            on_threshold: Window score at or above which a track becomes WEARING.
            off_threshold: Window score at or below which a track becomes NOT_WEARING.
        """
        self.evict_after_frames = evict_after_frames
        self.window_s = window_s
        self.min_observations = min_observations
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.states: dict[int, GoggleState] = {}

    def update(self, timestamp: float, scores_by_id: dict[int, float]) -> dict[int, GoggleState]:
        """
        Adds one frame of observations and updates every track's state.

        Params:
            timestamp: Time the frame was observed, in seconds (monotonic).
            scores_by_id: score_frame output for each track seen in this frame.
        Returns:
            The current states, keyed by track ID.
        """
        for track_id, goggle_state in self.states.items():
            if track_id not in scores_by_id:
                goggle_state.frames_since_last_face += 1

        for track_id, score in scores_by_id.items():
            goggle_state = self.states.setdefault(track_id, GoggleState(track_id=track_id))
            goggle_state.frames_since_last_face = 0
            goggle_state.observations.append((timestamp, score))
            self._update_state(goggle_state, timestamp)

        self.states = {
            track_id: goggle_state
            for track_id, goggle_state in self.states.items()
            if goggle_state.frames_since_last_face <= self.evict_after_frames
        }
        return self.states

    def _update_state(self, goggle_state: GoggleState, timestamp: float) -> None:
        """Drops stale observations and applies hysteresis to the window score."""
        while goggle_state.observations and timestamp - goggle_state.observations[0][0] > self.window_s:
            goggle_state.observations.popleft()

        # Too little evidence keeps the current state, which is UNKNOWN for new tracks.
        if len(goggle_state.observations) < self.min_observations:
            return

        score = goggle_state.window_score
        if goggle_state.state != WearState.WEARING and score >= self.on_threshold:
            goggle_state.state = WearState.WEARING
        elif goggle_state.state != WearState.NOT_WEARING and score <= self.off_threshold:
            goggle_state.state = WearState.NOT_WEARING
