from __future__ import annotations


def build_translation_user_message(
    *, text: str, context: str, scene_participant_count: int | None = None
) -> str:
    input_block = f"<input>\n{text}\n</input>"
    if context:
        user_message = f"<context>\n{context}\n</context>\n\n{input_block}"
    else:
        user_message = input_block
    if scene_participant_count is None:
        return user_message
    if isinstance(scene_participant_count, bool):
        return user_message
    if not isinstance(scene_participant_count, int):
        return user_message
    if scene_participant_count < 1:
        return user_message
    return f"<scene>\nPeople: {scene_participant_count}\n</scene>\n\n{user_message}"
