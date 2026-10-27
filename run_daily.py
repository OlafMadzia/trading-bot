"""Entry point for the scheduled daily run."""
import traceback
import warnings

warnings.filterwarnings("ignore")

from bot.live import log, run_daily  # noqa: E402

if __name__ == "__main__":
    try:
        run_daily()
    except Exception:
        log("FATAL\n" + traceback.format_exc())
        raise
