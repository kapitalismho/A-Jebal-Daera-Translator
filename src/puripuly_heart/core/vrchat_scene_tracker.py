from __future__ import annotations

from dataclasses import dataclass, field

from puripuly_heart.core.vrchat_scene import SceneStatus, VrchatSceneSnapshot
from puripuly_heart.core.vrchat_scene_events import (
    DEGRADED_SNAPSHOT,
    SYNCING_SNAPSHOT,
    UNAVAILABLE_SNAPSHOT,
    IdlessPresence,
    InstanceTransition,
    LeftRoom,
    LocalPlayerInitialized,
    PlayerJoined,
    PlayerLeft,
    VrchatSceneEvent,
    ready_snapshot,
)


@dataclass(slots=True)
class VrchatScenePresenceTracker:
    _members: set[str] = field(default_factory=set, init=False, repr=False)
    _status: SceneStatus = field(default="syncing", init=False)
    _local_initialized: bool = field(default=False, init=False)
    _leaves_suppressed: bool = field(default=False, init=False)

    @property
    def status(self) -> SceneStatus:
        return self._status

    @property
    def member_count(self) -> int:
        return len(self._members)

    @property
    def local_initialized(self) -> bool:
        return self._local_initialized

    def reset(self) -> None:
        self._members.clear()
        self._status = "syncing"
        self._local_initialized = False
        self._leaves_suppressed = False

    def on_event(self, event: VrchatSceneEvent) -> None:
        if isinstance(event, InstanceTransition):
            self.on_transition()
        elif isinstance(event, LeftRoom):
            self.on_left_room()
        elif isinstance(event, PlayerJoined):
            self.on_join(event.user_id)
        elif isinstance(event, PlayerLeft):
            self.on_leave(event.user_id)
        elif isinstance(event, LocalPlayerInitialized):
            self.on_local_initialized()
        elif isinstance(event, IdlessPresence):
            self.on_idless_presence()

    def on_transition(self) -> None:
        self._members.clear()
        self._status = "syncing"
        self._local_initialized = False
        self._leaves_suppressed = False

    def on_left_room(self) -> None:
        self._members.clear()
        self._status = "unavailable"
        self._local_initialized = False
        self._leaves_suppressed = True

    def on_join(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id:
            self.on_idless_presence()
            return
        if self._status == "unavailable" or self._leaves_suppressed:
            self._members.clear()
            self._local_initialized = False
            self._leaves_suppressed = False
            self._status = "syncing"
        if self._status == "degraded":
            return
        self._members.add(user_id)

    def on_leave(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id:
            self.on_idless_presence()
            return
        if self._leaves_suppressed or self._status == "unavailable":
            return
        if self._status == "degraded":
            return
        if user_id not in self._members:
            self._status = "degraded"
            return
        self._members.discard(user_id)
        if not self._members:
            self._status = "degraded"

    def on_local_initialized(self) -> None:
        if self._leaves_suppressed or self._status == "unavailable":
            self._leaves_suppressed = False
            self._status = "syncing"
        if self._status == "degraded":
            return
        self._local_initialized = True

    def on_idless_presence(self) -> None:
        if self._leaves_suppressed or self._status == "unavailable":
            return
        if self._status == "degraded":
            return
        self._status = "degraded"

    def on_file_failure(self) -> None:
        if self._status == "unavailable":
            return
        self._status = "degraded"

    def settle_if_quiet(self) -> None:
        if self._status != "syncing":
            return
        if not self._members or not self._local_initialized:
            return
        self._status = "ready"

    def snapshot(self) -> VrchatSceneSnapshot:
        if self._status == "ready":
            if not self._members:
                return DEGRADED_SNAPSHOT
            return ready_snapshot(len(self._members))
        if self._status == "syncing":
            return SYNCING_SNAPSHOT
        if self._status == "degraded":
            return DEGRADED_SNAPSHOT
        return UNAVAILABLE_SNAPSHOT


__all__ = ["VrchatScenePresenceTracker"]
