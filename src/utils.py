# =========================================================
#  src/utils.py
# =========================================================
import os
from datetime import datetime

def get_timestamp():
    """Return formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log(message, log_file="../outputs/logs/training_log.txt"):
    """
    Append message to the log file.
    Creates folder automatically if missing.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"[{get_timestamp()}] {message}\n")
