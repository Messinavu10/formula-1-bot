import os
import sys
import logging
from datetime import datetime

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create timestamp for unique log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"f1_pipeline_{timestamp}.log"
log_filepath = os.path.join(log_dir, log_filename)

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("f1logger")

# Log the start of a new run
logger.info(f"=== NEW PIPELINE RUN STARTED ===")
logger.info(f"Log file: {log_filename}")
logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"=================================")