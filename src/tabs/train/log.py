import threading
import time
from collections.abc import Callable
from pathlib import Path

LOG_POLL_INTERVAL = 0.5


def monitor_log_with_progress(
    log_file: Path,
    done_event: threading.Event,
    progress_callback: Callable[[str], None],
    poll_interval: float = LOG_POLL_INTERVAL,
) -> str:
    """Monitor a log file and update progress until completion."""
    while not done_event.is_set():
        progress_callback(log_file.read_text())
        time.sleep(poll_interval)
    return log_file.read_text()
