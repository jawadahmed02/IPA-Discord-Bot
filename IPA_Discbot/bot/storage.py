import sqlite3
from datetime import UTC, datetime
import json

from cryptography.fernet import Fernet, InvalidToken

from .config import BOT_MASTER_KEY, BOT_SESSION_ID, DB_PATH


def _get_fernet() -> Fernet:
    if not BOT_MASTER_KEY:
        raise RuntimeError("Missing BOT_MASTER_KEY in environment")
    try:
        return Fernet(BOT_MASTER_KEY.strip().encode())
    except Exception as e:
        raise RuntimeError("BOT_MASTER_KEY is invalid") from e


def encrypt_api_key(api_key: str) -> bytes:
    return _get_fernet().encrypt(api_key.encode("utf-8"))


def decrypt_api_key(token: bytes) -> str:
    try:
        return _get_fernet().decrypt(token).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError(
            "Failed to decrypt API key: invalid master key or corrupted data"
        ) from e


def _db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def migrate_plaintext_keys_if_needed():
    con = _db_connect()
    try:
        cur = con.cursor()

        cur.execute("PRAGMA table_info(user_provider_keys)")
        columns = cur.fetchall()
        column_names = {col[1] for col in columns}

        if "api_key" in column_names:
            print("[DB] Migrating plaintext API keys to encrypted storage...")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_provider_keys_new (
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    api_key_encrypted BLOB NOT NULL,
                    PRIMARY KEY (user_id, provider)
                )
                """
            )

            cur.execute("SELECT user_id, provider, api_key FROM user_provider_keys")
            rows = cur.fetchall()

            for user_id, provider, plaintext_key in rows:
                if plaintext_key is None:
                    continue
                plaintext_key = str(plaintext_key).strip()
                if not plaintext_key:
                    continue

                encrypted = encrypt_api_key(plaintext_key)
                cur.execute(
                    """
                    INSERT OR REPLACE INTO user_provider_keys_new (user_id, provider, api_key_encrypted)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, provider, encrypted),
                )

            cur.execute("DROP TABLE user_provider_keys")
            cur.execute("ALTER TABLE user_provider_keys_new RENAME TO user_provider_keys")

            con.commit()
            print("[DB] Migration complete.")
        else:
            con.commit()
    finally:
        con.close()


def migrate_messages_session_column_if_needed():
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(messages)")
        columns = cur.fetchall()
        column_names = {col[1] for col in columns}

        if "session_id" not in column_names:
            print("[DB] Adding session_id column to messages...")
            cur.execute("ALTER TABLE messages ADD COLUMN session_id TEXT")

        cur.execute(
            """
            UPDATE messages
            SET session_id = 'legacy'
            WHERE session_id IS NULL OR TRIM(session_id) = ''
            """
        )
        con.commit()
    finally:
        con.close()


def init_db():
    con = _db_connect()
    cur = con.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            guild_id TEXT,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_provider_keys (
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            api_key_encrypted BLOB NOT NULL,
            PRIMARY KEY (user_id, provider)
        );

        CREATE TABLE IF NOT EXISTS user_model_selection (
            user_id TEXT NOT NULL PRIMARY KEY,
            model_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_share_mode (
            user_id TEXT NOT NULL PRIMARY KEY,
            share_mode TEXT NOT NULL CHECK (share_mode IN ('individual', 'group'))
        );

        CREATE TABLE IF NOT EXISTS user_channel_chat_mode (
            user_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            chat_enabled INTEGER NOT NULL CHECK (chat_enabled IN (0, 1)),
            PRIMARY KEY (user_id, channel_id)
        );

        CREATE TABLE IF NOT EXISTS saved_sessions (
            session_id TEXT PRIMARY KEY,
            saved_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS saved_working_artifacts (
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            artifacts_json TEXT NOT NULL,
            history_json TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            PRIMARY KEY (channel_id, user_id)
        );
    """
    )

    con.commit()
    con.close()

    migrate_plaintext_keys_if_needed()
    migrate_messages_session_column_if_needed()
    cleanup_unsaved_sessions()
    print(f"[DB] Bot session started: {BOT_SESSION_ID}")


def log_message(
    channel_id: int,
    user_id: int,
    role: str,
    content: str,
    guild_id: int | None,
):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO messages (ts, guild_id, channel_id, user_id, session_id, role, content)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            str(guild_id) if guild_id else None,
            str(channel_id),
            str(user_id),
            BOT_SESSION_ID,
            role,
            content,
        ),
    )
    con.commit()
    con.close()


def cleanup_unsaved_sessions():
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            DELETE FROM messages
            WHERE session_id IS NOT NULL
              AND TRIM(session_id) != ''
              AND session_id != 'legacy'
              AND session_id NOT IN (SELECT session_id FROM saved_sessions)
            """
        )
        deleted_rows = cur.rowcount if cur.rowcount is not None else 0

        cur.execute(
            """
            DELETE FROM saved_working_artifacts
            WHERE session_id IS NOT NULL
              AND TRIM(session_id) != ''
              AND session_id NOT IN (SELECT session_id FROM saved_sessions)
            """
        )
        deleted_artifact_rows = cur.rowcount if cur.rowcount is not None else 0

        cur.execute(
            """
            DELETE FROM saved_sessions
            WHERE session_id NOT IN (SELECT DISTINCT session_id FROM messages)
            """
        )
        con.commit()
    finally:
        con.close()

    if deleted_rows:
        print(f"[DB] Deleted {deleted_rows} unsaved message(s) from prior bot sessions.")
    if deleted_artifact_rows:
        print(f"[DB] Deleted {deleted_artifact_rows} unsaved artifact snapshot(s) from prior bot sessions.")


def save_current_session() -> bool:
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT 1 FROM saved_sessions WHERE session_id = ?",
            (BOT_SESSION_ID,),
        )
        already_saved = cur.fetchone() is not None
        if not already_saved:
            cur.execute(
                """
                INSERT INTO saved_sessions (session_id, saved_at)
                VALUES (?, ?)
                """,
                (BOT_SESSION_ID, datetime.now(UTC).isoformat()),
            )
            con.commit()
        return not already_saved
    finally:
        con.close()


def is_session_saved(session_id: str) -> bool:
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT 1 FROM saved_sessions WHERE session_id = ?",
            (session_id,),
        )
        return cur.fetchone() is not None
    finally:
        con.close()


def save_working_artifacts_snapshot(
    session_id: str,
    artifacts_by_key: dict[tuple[int, int], dict[str, str]],
    history_by_key: dict[tuple[int, int], list[dict[str, str]]],
):
    con = _db_connect()
    try:
        cur = con.cursor()
        saved_at = datetime.now(UTC).isoformat()
        for (channel_id, user_id), artifacts in artifacts_by_key.items():
            artifact_payload = {str(k): str(v) for k, v in dict(artifacts).items()}
            history_payload = [
                {str(k): str(v) for k, v in dict(snapshot).items()}
                for snapshot in history_by_key.get((channel_id, user_id), [])
            ]
            cur.execute(
                """
                INSERT INTO saved_working_artifacts (
                    channel_id, user_id, session_id, artifacts_json, history_json, saved_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id, user_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    artifacts_json=excluded.artifacts_json,
                    history_json=excluded.history_json,
                    saved_at=excluded.saved_at
                """,
                (
                    str(channel_id),
                    str(user_id),
                    session_id,
                    json.dumps(artifact_payload),
                    json.dumps(history_payload),
                    saved_at,
                ),
            )
        con.commit()
    finally:
        con.close()


def load_saved_working_artifacts() -> tuple[dict[tuple[int, int], dict[str, str]], dict[tuple[int, int], list[dict[str, str]]]]:
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT channel_id, user_id, artifacts_json, history_json
            FROM saved_working_artifacts
            ORDER BY saved_at ASC
            """
        )
        rows = cur.fetchall()
    finally:
        con.close()

    artifacts: dict[tuple[int, int], dict[str, str]] = {}
    history: dict[tuple[int, int], list[dict[str, str]]] = {}

    for channel_id, user_id, artifacts_json, history_json in rows:
        try:
            artifact_payload = json.loads(artifacts_json or "{}")
        except json.JSONDecodeError:
            artifact_payload = {}
        try:
            history_payload = json.loads(history_json or "[]")
        except json.JSONDecodeError:
            history_payload = []

        key = (int(channel_id), int(user_id))
        artifacts[key] = {
            str(k): str(v) for k, v in artifact_payload.items() if v is not None
        }
        history[key] = [
            {str(k): str(v) for k, v in snapshot.items() if v is not None}
            for snapshot in history_payload
            if isinstance(snapshot, dict)
        ]

    return artifacts, history


def get_recent_context(
    user_id: int,
    guild_id: int | None,
    channel_id: int | None = None,
    limit: int = 1000,
):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content, session_id, channel_id
        FROM messages
        WHERE user_id = ?
          AND (
                (? IS NULL AND guild_id IS NULL)
                OR guild_id = ?
              )
        ORDER BY
            CASE WHEN channel_id = ? THEN 0 ELSE 1 END,
            id DESC
        LIMIT ?
        """,
        (
            str(user_id),
            str(guild_id) if guild_id is not None else None,
            str(guild_id) if guild_id is not None else None,
            str(channel_id) if channel_id is not None else None,
            limit,
        ),
    )
    rows = cur.fetchall()
    con.close()
    rows.reverse()

    context: list[dict] = []
    previous_session_id: str | None = None
    previous_channel_id: str | None = None

    for role, content, session_id, row_channel_id in rows:
        if session_id != previous_session_id:
            context.append(
                {"role": "system", "content": f"Conversation session: {session_id}"}
            )
            previous_session_id = session_id

        if row_channel_id != previous_channel_id:
            context.append(
                {"role": "system", "content": f"Discord channel: {row_channel_id}"}
            )
            previous_channel_id = row_channel_id

        context.append({"role": role, "content": content})

    return context


def get_user_model(user_id: str) -> str | None:
    con = _db_connect()
    cur = con.cursor()
    cur.execute("SELECT model_id FROM user_model_selection WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def set_user_model(user_id: str, model_id: str):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_model_selection (user_id, model_id)
        VALUES (?, ?) ON CONFLICT(user_id) DO
        UPDATE SET model_id=excluded.model_id
        """,
        (user_id, model_id),
    )
    con.commit()
    con.close()


def save_provider_key(user_id: str, provider: str, api_key: str):
    encrypted_key = encrypt_api_key(api_key.strip())

    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_provider_keys (user_id, provider, api_key_encrypted)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, provider) DO UPDATE SET api_key_encrypted=excluded.api_key_encrypted
        """,
        (user_id, provider, encrypted_key),
    )
    con.commit()
    con.close()


def user_has_any_provider_key(user_id: str) -> bool:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT 1 FROM user_provider_keys WHERE user_id = ? LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    con.close()
    return row is not None


def get_provider_key(user_id: str, provider: str) -> str | None:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT api_key_encrypted FROM user_provider_keys WHERE user_id = ? AND provider = ?",
        (user_id, provider),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    encrypted_key = row[0]
    return decrypt_api_key(encrypted_key)


def get_share_mode(user_id: str) -> str:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT share_mode FROM user_share_mode WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else "individual"


def set_share_mode(user_id: str, share_mode: str):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_share_mode (user_id, share_mode)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET share_mode=excluded.share_mode
        """,
        (user_id, share_mode),
    )
    con.commit()
    con.close()


def is_chat_enabled(user_id: str, channel_id: str) -> bool:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT chat_enabled FROM user_channel_chat_mode WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    row = cur.fetchone()
    con.close()
    return bool(row[0]) if row else True


def set_chat_enabled(user_id: str, channel_id: str, enabled: bool):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_channel_chat_mode (user_id, channel_id, chat_enabled)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, channel_id) DO UPDATE SET chat_enabled=excluded.chat_enabled
        """,
        (user_id, channel_id, int(enabled)),
    )
    con.commit()
    con.close()


def get_effective_provider_key(user_id: str, provider: str) -> tuple[str | None, str | None]:
    own_key = get_provider_key(user_id, provider)
    if own_key:
        return own_key, user_id

    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT upk.user_id, upk.api_key_encrypted
        FROM user_provider_keys upk
        JOIN user_share_mode usm ON usm.user_id = upk.user_id
        WHERE upk.provider = ?
          AND usm.share_mode = 'group'
          AND upk.user_id != ?
        ORDER BY upk.user_id
        LIMIT 1
        """,
        (provider, user_id),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return None, None

    owner_user_id, encrypted_key = row
    return decrypt_api_key(encrypted_key), owner_user_id
