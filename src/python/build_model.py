# Packages
from utils import DiagnosisSegmentModel

# Constants
DB_CONFIG = "../../configs/db_config.json"
MODEL_CONFIG = "../../configs/model_config.json"

# Build Model
dm = DiagnosisSegmentModel(db_config=DB_CONFIG, model_config=MODEL_CONFIG)
dm.build_model()