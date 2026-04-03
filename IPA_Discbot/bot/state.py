import discord

from .config import ARTIFACT_HISTORY, BOT_SESSION_ID, LAST_SOLVE_ARTIFACTS
from .storage import is_collab_enabled, save_working_artifacts_snapshot

SHARED_ARTIFACT_OWNER_ID = 0


def _solve_artifacts_key(message: discord.Message) -> tuple[int, int]:
    if is_collab_enabled(str(message.channel.id)):
        return (message.channel.id, SHARED_ARTIFACT_OWNER_ID)
    return (message.channel.id, message.author.id)


def _latest_artifact_key_for_user(user_id: int) -> tuple[int, int] | None:
    for key in reversed(list(LAST_SOLVE_ARTIFACTS.keys())):
        if key[1] != user_id:
            continue
        artifacts = LAST_SOLVE_ARTIFACTS.get(key, {})
        if any(str(artifacts.get(name, "")).strip() for name in ("domain", "problem", "plan")):
            return key
    return None


def _working_artifacts(message: discord.Message) -> dict[str, str]:
    key = _solve_artifacts_key(message)
    current = LAST_SOLVE_ARTIFACTS.get(key)
    if current is not None:
        return current

    if key[1] == SHARED_ARTIFACT_OWNER_ID:
        return LAST_SOLVE_ARTIFACTS.setdefault(key, {})

    fallback_key = _latest_artifact_key_for_user(message.author.id)
    if fallback_key is not None and fallback_key != key:
        LAST_SOLVE_ARTIFACTS[key] = {
            k: str(v) for k, v in LAST_SOLVE_ARTIFACTS.get(fallback_key, {}).items()
        }
        fallback_history = ARTIFACT_HISTORY.get(fallback_key, [])
        if fallback_history:
            ARTIFACT_HISTORY[key] = [
                {k: str(v) for k, v in snapshot.items()} for snapshot in fallback_history
            ]
        return LAST_SOLVE_ARTIFACTS[key]

    return LAST_SOLVE_ARTIFACTS.setdefault(key, {})


def _persist_artifacts_if_session_saved() -> None:
    save_working_artifacts_snapshot(
        BOT_SESSION_ID,
        LAST_SOLVE_ARTIFACTS,
        ARTIFACT_HISTORY,
    )


def _push_artifact_history(message: discord.Message) -> None:
    key = _solve_artifacts_key(message)
    current = LAST_SOLVE_ARTIFACTS.get(key)
    if not current:
        return
    snapshot = {k: str(v) for k, v in current.items()}
    history = ARTIFACT_HISTORY.setdefault(key, [])
    history.append(snapshot)
    if len(history) > 10:
        del history[:-10]


def _update_working_artifacts(message: discord.Message, **updates: str) -> dict[str, str]:
    current = _working_artifacts(message)
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        value_str = str(value)
        if current.get(key) != value_str:
            changed = True
    if changed and current:
        _push_artifact_history(message)
    current.update({k: str(v) for k, v in updates.items() if v is not None})
    if changed:
        _persist_artifacts_if_session_saved()
    return current


def _artifact_text(current: dict[str, str], artifact_type: str) -> str:
    return str(current.get(artifact_type, "")).strip()


def _store_last_solve_artifacts(
    message: discord.Message,
    domain_text: str,
    problem_text: str,
    plan_text: str,
    domain_name: str,
    problem_name: str,
    **extra_artifacts: str,
) -> None:
    artifacts = {
        "domain": domain_text,
        "problem": problem_text,
        "plan": plan_text,
        "domain_name": domain_name,
        "problem_name": problem_name,
    }
    artifacts.update({k: str(v) for k, v in extra_artifacts.items() if v is not None})
    LAST_SOLVE_ARTIFACTS[_solve_artifacts_key(message)] = artifacts
    _persist_artifacts_if_session_saved()


def _restore_artifact_version(message: discord.Message, artifact_type: str) -> dict[str, str]:
    key = _solve_artifacts_key(message)
    history = ARTIFACT_HISTORY.get(key, [])
    if not history:
        raise RuntimeError("No previous artifact version to restore.")

    current = _working_artifacts(message)
    restored_index = None
    for index in range(len(history) - 1, -1, -1):
        snapshot = history[index]
        if _artifact_text(snapshot, artifact_type):
            restored_index = index
            current.update(snapshot)
            del history[index:]
            _persist_artifacts_if_session_saved()
            break
    if restored_index is None:
        raise RuntimeError(f"No previous {artifact_type} version to restore.")

    return current


def _copy_personal_artifacts_to_shared(channel_id: int, user_id: int) -> None:
    personal_key = (channel_id, user_id)
    shared_key = (channel_id, SHARED_ARTIFACT_OWNER_ID)
    if shared_key not in LAST_SOLVE_ARTIFACTS and personal_key in LAST_SOLVE_ARTIFACTS:
        LAST_SOLVE_ARTIFACTS[shared_key] = {
            k: str(v) for k, v in LAST_SOLVE_ARTIFACTS[personal_key].items()
        }
    if shared_key not in ARTIFACT_HISTORY and personal_key in ARTIFACT_HISTORY:
        ARTIFACT_HISTORY[shared_key] = [
            {k: str(v) for k, v in snapshot.items()}
            for snapshot in ARTIFACT_HISTORY[personal_key]
        ]
    _persist_artifacts_if_session_saved()


def _load_runtime_artifact_state(
    saved_artifacts: dict[tuple[int, int], dict[str, str]],
    saved_history: dict[tuple[int, int], list[dict[str, str]]],
) -> None:
    LAST_SOLVE_ARTIFACTS.clear()
    LAST_SOLVE_ARTIFACTS.update(saved_artifacts)
    ARTIFACT_HISTORY.clear()
    ARTIFACT_HISTORY.update(saved_history)
