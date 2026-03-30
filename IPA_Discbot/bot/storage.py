import sqlite3
from datetime import UTC, datetime

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
    """
    )

    con.commit()
    con.close()

    migrate_plaintext_keys_if_needed()
    migrate_messages_session_column_if_needed()
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
