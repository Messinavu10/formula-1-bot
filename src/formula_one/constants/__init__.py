import os
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Configuration files
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")
PARAMS_FILE_PATH = os.path.join(ROOT_DIR, "params.yaml")

# Artifacts directory
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Logs directory
LOGS_DIR = os.path.join(ROOT_DIR, "artifacts", "logs")