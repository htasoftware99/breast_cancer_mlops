# import os
# import pandas as pd
# import numpy as np
# import pickle
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from utils.common_functions import read_yaml, load_data
# from sklearn.preprocessing import StandardScaler

# logger = get_logger(__name__)

# class DataProcessor:

#     def __init__(self, train_path, test_path, processed_dir, config_path):
#         self.train_path = train_path
#         self.test_path = test_path
#         self.processed_dir = processed_dir
#         self.config = read_yaml(config_path)

#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)

#     def preprocess_data(self, df):
#         try:
#             logger.info("Starting data preprocessing step")

#             cols_to_drop = [c for c in ['Unnamed: 0', 'id'] if c in df.columns]
#             df.drop(columns=cols_to_drop, inplace=True)
#             logger.info(f"Dropped columns: {cols_to_drop}")

#             df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
#             logger.info("Label encoding applied: M=1, B=0")

#             return df

#         except Exception as e:
#             logger.error(f"Error during preprocess step: {e}")
#             raise CustomException("Error while preprocess data", e)

#     def remove_high_correlation(self, df):
#         try:
#             threshold = self.config["data_processing"]["correlation_threshold"]
#             logger.info(f"Removing features with correlation > {threshold}")

#             features = df.drop(columns=['diagnosis'])
#             corr_matrix = features.corr().abs()

#             mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#             tri_df = corr_matrix.mask(mask)

#             to_drop = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
#             df.drop(columns=to_drop, inplace=True)

#             logger.info(f"Dropped {len(to_drop)} correlated features: {to_drop}")
#             logger.info(f"Remaining features: {df.shape[1] - 1}")

#             return df, to_drop  # to_drop'u test setine de uygulamak için döndür

#         except Exception as e:
#             logger.error(f"Error during correlation removal: {e}")
#             raise CustomException("Error while removing high correlation features", e)

#     def scale_data(self, train_df, test_df):
#         try:
#             logger.info("Applying StandardScaler")

#             X_train = train_df.drop(columns=['diagnosis'])
#             y_train = train_df['diagnosis']

#             X_test = test_df.drop(columns=['diagnosis'])
#             y_test = test_df['diagnosis']

#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)   # fit sadece train'de
#             X_test_scaled  = scaler.transform(X_test)        # test'e sadece transform

#             # Scaler'ı kaydet
#             scaler_path = os.path.join(self.processed_dir, 'scaler.pkl')
#             with open(scaler_path, 'wb') as f:
#                 pickle.dump(scaler, f)
#             logger.info(f"Scaler saved to {scaler_path}")

#             train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#             train_scaled_df['diagnosis'] = y_train.reset_index(drop=True)

#             test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#             test_scaled_df['diagnosis'] = y_test.reset_index(drop=True)

#             return train_scaled_df, test_scaled_df

#         except Exception as e:
#             logger.error(f"Error during scaling: {e}")
#             raise CustomException("Error while scaling data", e)

#     def save_data(self, df, file_path):
#         try:
#             logger.info(f"Saving data to {file_path}")
#             df.to_csv(file_path, index=False)
#             logger.info("Data saved successfully")

#         except Exception as e:
#             logger.error(f"Error during saving data: {e}")
#             raise CustomException("Error while saving data", e)

#     def process(self):
#         try:
#             logger.info("Loading data from RAW directory")
#             train_df = load_data(self.train_path)
#             test_df  = load_data(self.test_path)

#             train_df = self.preprocess_data(train_df)
#             test_df  = self.preprocess_data(test_df)

#             train_df, to_drop = self.remove_high_correlation(train_df)
#             test_df.drop(columns=[c for c in to_drop if c in test_df.columns], inplace=True)

#             train_df, test_df = self.scale_data(train_df, test_df)

#             self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
#             self.save_data(test_df,  PROCESSED_TEST_DATA_PATH)

#             logger.info("Data processing pipeline completed successfully")

#         except Exception as e:
#             logger.error(f"Error in processing pipeline: {e}")
#             raise CustomException("Error in data processing pipeline", e)


# if __name__ == "__main__":
#     processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
#     processor.process()











import os
import pandas as pd
import numpy as np
import pickle
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)


class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path, feature_store: RedisFeatureStore = None):
        self.train_path    = train_path
        self.test_path     = test_path
        self.processed_dir = processed_dir
        self.config        = read_yaml(config_path)
        self.feature_store = feature_store          # Optional — if None, Redis step is skipped

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        logger.info("DataProcessor initialized.")

    # ─────────────────────────────────────────────────────────────────
    def preprocess_data(self, df):
        try:
            logger.info("Preprocessing started.")

            cols_to_drop = [c for c in ['Unnamed: 0', 'id'] if c in df.columns]
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped columns: {cols_to_drop}")

            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            logger.info("Label encoding applied: M=1, B=0")

            return df

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise CustomException("Error during preprocessing", e)

    # ─────────────────────────────────────────────────────────────────
    def remove_high_correlation(self, df):
        try:
            threshold = self.config["data_processing"]["correlation_threshold"]
            logger.info(f"Correlation threshold: {threshold}")

            features     = df.drop(columns=['diagnosis'])
            corr_matrix  = features.corr().abs()
            mask         = np.triu(np.ones_like(corr_matrix, dtype=bool))
            tri_df       = corr_matrix.mask(mask)
            to_drop      = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

            df.drop(columns=to_drop, inplace=True)

            logger.info(f"Dropped {len(to_drop)} correlated features: {to_drop}")
            logger.info(f"Remaining features: {df.shape[1] - 1}")

            return df, to_drop

        except Exception as e:
            logger.error(f"Correlation removal error: {e}")
            raise CustomException("Error removing high-correlation features", e)

    # ─────────────────────────────────────────────────────────────────
    def scale_data(self, train_df, test_df):
        try:
            logger.info("Applying StandardScaler...")

            X_train = train_df.drop(columns=['diagnosis'])
            y_train = train_df['diagnosis']
            X_test  = test_df.drop(columns=['diagnosis'])
            y_test  = test_df['diagnosis']

            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)   # Fit on train only
            X_test_scaled  = scaler.transform(X_test)        # Transform test only

            # Save scaler for use in app.py
            scaler_path = os.path.join(self.processed_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")

            train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            train_scaled_df['diagnosis'] = y_train.reset_index(drop=True)

            test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            test_scaled_df['diagnosis'] = y_test.reset_index(drop=True)

            return train_scaled_df, test_scaled_df

        except Exception as e:
            logger.error(f"Scaling error: {e}")
            raise CustomException("Error during scaling", e)

    # ─────────────────────────────────────────────────────────────────
    def store_features_in_redis(self, train_df, test_df):
        """
        Writes UNSCALED train + test data to Redis Feature Store.
        Must be called BEFORE scale_data() so that the drift detector
        in app.py can fit its own scaler on the raw values — exactly
        like the reference implementation.
        diagnosis column is excluded since it is not a feature.
        """
        try:
            batch_data = {}

            for label, df in [("train", train_df), ("test", test_df)]:
                feature_cols = [c for c in df.columns if c != "diagnosis"]
                for idx, row in df[feature_cols].iterrows():
                    entity_id = f"{label}_{idx}"
                    batch_data[entity_id] = row.to_dict()

            self.feature_store.store_batch_features(batch_data)
            logger.info(f"{len(batch_data)} unscaled records written to Redis.")

        except Exception as e:
            logger.error(f"Redis write error: {e}")
            raise CustomException("Failed to write to Feature Store", e)

    # ─────────────────────────────────────────────────────────────────
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully.")
        except Exception as e:
            logger.error(f"Save error: {e}")
            raise CustomException("Error saving data", e)

    # ─────────────────────────────────────────────────────────────────
    def process(self):
        try:
            logger.info("Data processing pipeline started.")

            train_df = load_data(self.train_path)
            test_df  = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df  = self.preprocess_data(test_df)

            train_df, to_drop = self.remove_high_correlation(train_df)
            test_df.drop(columns=[c for c in to_drop if c in test_df.columns], inplace=True)

            # Write to Redis BEFORE scaling — drift detector needs raw values
            if self.feature_store:
                self.store_features_in_redis(train_df, test_df)

            train_df, test_df = self.scale_data(train_df, test_df)

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,  PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed.")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise CustomException("Error in data processing pipeline", e)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feature_store = RedisFeatureStore()   # Redis must be running via Docker

    processor = DataProcessor(
        train_path    = TRAIN_FILE_PATH,
        test_path     = TEST_FILE_PATH,
        processed_dir = PROCESSED_DIR,
        config_path   = CONFIG_PATH,
        feature_store = feature_store     # Pass None to skip Redis step
    )
    processor.process()