from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from puripuly_heart.core.vrchat_scene import SceneStatus, VrchatSceneSnapshot


@dataclass(frozen=True, slots=True)
class InstanceTransition:
    kind: str = "transition"


@dataclass(frozen=True, slots=True)
class LeftRoom:
    kind: str = "left-room"


@dataclass(frozen=True, slots=True)
class PlayerJoined:
    user_id: str
    kind: str = "joined"

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("joined event requires a non-empty user id")


@dataclass(frozen=True, slots=True)
class PlayerLeft:
    user_id: str
    kind: str = "left"

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("left event requires a non-empty user id")


@dataclass(frozen=True, slots=True)
class LocalPlayerInitialized:
    kind: str = "local-initialized"


@dataclass(frozen=True, slots=True)
class IdlessPresence:
    kind: str = "idless-presence"


VrchatSceneEvent = Union[
    InstanceTransition,
    LeftRoom,
    PlayerJoined,
    PlayerLeft,
    LocalPlayerInitialized,
    IdlessPresence,
]

UNAVAILABLE_SNAPSHOT = VrchatSceneSnapshot(status="unavailable")

SYNCING_SNAPSHOT = VrchatSceneSnapshot(status="syncing")

DEGRADED_SNAPSHOT = VrchatSceneSnapshot(status="degraded")


def ready_snapshot(participant_count: int) -> VrchatSceneSnapshot:
    return VrchatSceneSnapshot(status="ready", participant_count=participant_count)


def status_snapshot(status: SceneStatus) -> VrchatSceneSnapshot:
    if status == "ready":
        raise ValueError("ready snapshots require ready_snapshot with a participant count")
    return VrchatSceneSnapshot(status=status)


__all__ = [
    "DEGRADED_SNAPSHOT",
    "IdlessPresence",
    "InstanceTransition",
    "LeftRoom",
    "LocalPlayerInitialized",
    "PlayerJoined",
    "PlayerLeft",
    "SYNCING_SNAPSHOT",
    "UNAVAILABLE_SNAPSHOT",
    "VrchatSceneEvent",
    "ready_snapshot",
    "status_snapshot",
]
