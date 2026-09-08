from __future__ import annotations

import re

from puripuly_heart.core.vrchat_scene_events import (
    IdlessPresence,
    InstanceTransition,
    LeftRoom,
    LocalPlayerInitialized,
    PlayerJoined,
    PlayerLeft,
    VrchatSceneEvent,
)

_ENVELOPE = re.compile(
    r"^(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2} (?:Log|Warning|Error|Debug)\s+-\s+)\[Behaviour\] ?"
)

_JOINED_VERB = "OnPlayerJoined "
_LEFT_VERB = "OnPlayerLeft "
_LEFT_ROOM_VERB = "OnLeftRoom"
_LEFT_ROOM_NOTICE_VERB = "OnPlayerLeftRoom"
_ENTERING_VERB = "Entering Room:"
_JOINING_WORLD_VERB = "Joining wrld_"
_INITIALIZED_PREFIX = 'Initialized PlayerAPI "'
_IS_LOCAL_SUFFIX = '" is local'
_USER_ID = re.compile(r"^usr_[A-Za-z0-9\-]+$")


def parse_vrchat_scene_line(line: str) -> VrchatSceneEvent | None:
    if not isinstance(line, str) or not line:
        return None
    envelope = _ENVELOPE.match(line)
    if envelope is None:
        return None
    payload = line[envelope.end() :]
    if payload.startswith(_ENTERING_VERB) or payload.startswith(_JOINING_WORLD_VERB):
        return InstanceTransition()
    if payload == _LEFT_ROOM_VERB:
        return LeftRoom()
    if payload == _LEFT_ROOM_NOTICE_VERB:
        return None
    if payload.startswith(_JOINED_VERB):
        user_id = _user_id_after(payload, len(_JOINED_VERB))
        if user_id is None:
            return IdlessPresence()
        return PlayerJoined(user_id=user_id)
    if payload.startswith(_LEFT_VERB):
        user_id = _user_id_after(payload, len(_LEFT_VERB))
        if user_id is None:
            return IdlessPresence()
        return PlayerLeft(user_id=user_id)
    if payload.startswith(_INITIALIZED_PREFIX) and payload.endswith(_IS_LOCAL_SUFFIX):
        return LocalPlayerInitialized()
    return None


def _user_id_after(payload: str, verb_end: int) -> str | None:
    remainder = payload[verb_end:]
    if not remainder.endswith(")"):
        return None
    start = remainder.rfind(" (")
    if start < 0:
        return None
    user_id = remainder[start + 2 : -1]
    if _USER_ID.match(user_id) is None:
        return None
    return user_id


__all__ = ["parse_vrchat_scene_line"]
