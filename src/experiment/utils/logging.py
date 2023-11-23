import datetime
import logging
import os
import pathlib

logger = logging.getLogger("AI Reports")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

current_time = datetime.datetime.now()
log_filename = f"run_{ current_time.strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Stream Handler for console output
logging_level = getattr(logging, os.getenv("LOG_LEVEL", "DEBUG"))
sh = logging.StreamHandler()
logger.setLevel(logging_level)
sh.setFormatter(formatter)
logger.addHandler(sh)

try:
    log_file = pathlib.Path(os.getenv("LOG_FILE_PATH")) / log_filename
except TypeError:
    log_file = False

# File Handler for file output
if log_file:
    logger.warning(f"Logging to file: {log_file}")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
