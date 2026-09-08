from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

SceneStatus = Literal["unavailable", "syncing", "ready", "degraded"]


@dataclass(frozen=True, slots=True)
class VrchatSceneSnapshot:
    status: SceneStatus = "unavailable"
    participant_count: int | None = None

    def __post_init__(self) -> None:
        if self.status == "ready":
            if isinstance(self.participant_count, bool) or not isinstance(
                self.participant_count, int
            ):
                raise TypeError("ready scene snapshot requires an integer participant count")
            if self.participant_count < 1:
                raise ValueError("ready scene snapshot requires a positive participant count")
            return
        if self.participant_count is not None:
            raise ValueError("only a ready scene snapshot may carry a participant count")


class SceneSnapshotProvider(Protocol):
    def snapshot(self) -> VrchatSceneSnapshot: ...


__all__ = ["SceneSnapshotProvider", "SceneStatus", "VrchatSceneSnapshot"]
