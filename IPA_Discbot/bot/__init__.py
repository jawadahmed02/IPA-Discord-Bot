from . import services as _services
from .config import BOT_MASTER_KEY, DISCORD_TOKEN, bot
from .storage import init_db


def run() -> None:
    if not DISCORD_TOKEN:
        raise RuntimeError("Missing DISCORD_TOKEN")
    if not BOT_MASTER_KEY:
        raise RuntimeError("Missing BOT_MASTER_KEY")

    init_db()
    bot.run(DISCORD_TOKEN)


__all__ = ["bot", "run"]
