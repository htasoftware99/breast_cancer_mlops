# from src.data_ingestion import DataIngestion
# from src.data_preprocessing import DataProcessor
# from src.model_training import ModelTraining
# from utils.common_functions import read_yaml
# from config.paths_config import *
# from src.logger import get_logger

# logger = get_logger(__name__)

# if __name__ == "__main__":

#     ### 1. Data Ingestion
#     logger.info("=" * 50)
#     logger.info("PIPELINE STARTED")
#     logger.info("=" * 50)

#     data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
#     data_ingestion.run()

#     ### 2. Data Processing
#     processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
#     processor.process()

#     ### 3. Model Training
#     trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
#     trainer.run()

#     logger.info("=" * 50)
#     logger.info("PIPELINE COMPLETED SUCCESSFULLY")
#     logger.info("=" * 50)









from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from src.feature_store import RedisFeatureStore
from utils.common_functions import read_yaml
from src.logger import get_logger
from config.paths_config import *

logger = get_logger(__name__)


if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("PIPELINE Started")
    logger.info("=" * 60)

    # ── Redis Feature Store Connection ────────────────────────────
    # If Docker is running, it will connect. If not, it will throw an error.
    # If you want to run it without Redis, set feature_store=None.
    try:
        feature_store = RedisFeatureStore(host="localhost", port=6379)
    except Exception:
        logger.warning("Could not connect to Redis! Continuing without the feature store.")
        feature_store = None

    # ── 1. Data Ingestion ─────────────────────────────────────────
    logger.info("Step 1: Data Ingestion")
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    # ── 2. Data Preprocessing ─────────────────────────────────────
    logger.info("Step 2: Data Preprocessing")
    processor = DataProcessor(
        train_path    = TRAIN_FILE_PATH,
        test_path     = TEST_FILE_PATH,
        processed_dir = PROCESSED_DIR,
        config_path   = CONFIG_PATH,
        feature_store = feature_store    # If Redis is available, it will write the processed data.
    )
    processor.process()

    # ── 3. Model Training ─────────────────────────────────────────
    logger.info("Step 3: Model Training")
    trainer = ModelTraining(
        train_path        = PROCESSED_TRAIN_DATA_PATH,
        test_path         = PROCESSED_TEST_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH,
        feature_store     = feature_store    # If Redis is available, it reads from there
    )
    trainer.run()

    logger.info("=" * 60)
    logger.info("PIPELINE Completed Successfully")
    logger.info("=" * 60)