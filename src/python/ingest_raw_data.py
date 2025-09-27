# Packages
from utils import IngestManager

# Constants
CONFIG = "../../configs/db_config.json"

# Run ingestion
im = IngestManager(config=CONFIG)
im.run_ingestion()