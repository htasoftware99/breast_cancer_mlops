# import os
# import joblib
# import mlflow
# import mlflow.sklearn
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from config.model_params import *
# from utils.common_functions import load_data

# logger = get_logger(__name__)

# # ─── MLflow settings ───────────────────────────────────────────────
# MLFLOW_EXPERIMENT_NAME = "Breast_Cancer_Logistic_Regression"
# # for local UI: mlflow ui  →  http://127.0.0.1:5000
# # mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")
# # ───────────────────────────────────────────────────────────────────


# class ModelTraining:

#     def __init__(self, train_path, test_path, model_output_path):
#         self.train_path       = train_path
#         self.test_path        = test_path
#         self.model_output_path = model_output_path
#         self.params_dist      = LOGISTIC_REGRESSION_PARAMS
#         self.grid_search_params = GRID_SEARCH_PARAMS

#     # ──────────────────────────────────────────────────────────────
#     def load_and_split_data(self):
#         try:
#             logger.info(f"Loading train data from {self.train_path}")
#             train_df = load_data(self.train_path)

#             logger.info(f"Loading test data from {self.test_path}")
#             test_df = load_data(self.test_path)

#             X_train = train_df.drop(columns=["diagnosis"])
#             y_train = train_df["diagnosis"]

#             X_test  = test_df.drop(columns=["diagnosis"])
#             y_test  = test_df["diagnosis"]

#             logger.info("Data loaded and split successfully")
#             return X_train, y_train, X_test, y_test

#         except Exception as e:
#             logger.error(f"Error while loading data: {e}")
#             raise CustomException("Failed to load data", e)

#     # ──────────────────────────────────────────────────────────────
#     def train_model(self, X_train, y_train):
#         try:
#             logger.info("Initializing Logistic Regression model")

#             lr_model = LogisticRegression(random_state=42)

#             logger.info("Starting hyperparameter tuning with GridSearchCV")
#             grid_search = GridSearchCV(
#                 estimator   = lr_model,
#                 param_grid  = self.params_dist,
#                 cv          = self.grid_search_params["cv"],
#                 n_jobs      = self.grid_search_params["n_jobs"],
#                 verbose     = self.grid_search_params["verbose"],
#                 scoring     = self.grid_search_params["scoring"]
#             )

#             grid_search.fit(X_train, y_train)

#             best_params = grid_search.best_params_
#             best_model  = grid_search.best_estimator_

#             logger.info(f"Best params: {best_params}")
#             return best_model, best_params

#         except Exception as e:
#             logger.error(f"Error while training model: {e}")
#             raise CustomException("Failed to train model", e)

#     # ──────────────────────────────────────────────────────────────
#     def evaluate_model(self, model, X_test, y_test):
#         try:
#             logger.info("Evaluating model")

#             y_pred  = model.predict(X_test)
#             y_proba = model.predict_proba(X_test)[:, 1]

#             metrics = {
#                 "accuracy"  : accuracy_score(y_test, y_pred),
#                 "precision" : precision_score(y_test, y_pred),
#                 "recall"    : recall_score(y_test, y_pred),
#                 "f1"        : f1_score(y_test, y_pred),
#                 "roc_auc"   : roc_auc_score(y_test, y_proba),
#             }

#             for k, v in metrics.items():
#                 logger.info(f"{k}: {v:.4f}")

#             return metrics

#         except Exception as e:
#             logger.error(f"Error while evaluating model: {e}")
#             raise CustomException("Failed to evaluate model", e)

#     # ──────────────────────────────────────────────────────────────
#     def save_model(self, model):
#         try:
#             os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
#             joblib.dump(model, self.model_output_path)
#             logger.info(f"Model saved to {self.model_output_path}")

#         except Exception as e:
#             logger.error(f"Error while saving model: {e}")
#             raise CustomException("Failed to save model", e)

#     # ──────────────────────────────────────────────────────────────
#     def run(self):
#         try:
#             logger.info("Starting Model Training pipeline")

#             # ── MLflow Experiment ──────────────────────────────────
#             mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

#             with mlflow.start_run(run_name="logistic_regression_gridsearch"):

#                 # 1) Load data
#                 X_train, y_train, X_test, y_test = self.load_and_split_data()

#                 # 2) Train model + hyperparameter tuning
#                 best_model, best_params = self.train_model(X_train, y_train)

#                 # 3) Evaluate model
#                 metrics = self.evaluate_model(best_model, X_test, y_test)

#                 # 4) MLflow → parameters
#                 mlflow.log_params(best_params)

#                 # log GridSearchCV parameters.
#                 mlflow.log_param("cv_folds",  self.grid_search_params["cv"])
#                 mlflow.log_param("scoring",   self.grid_search_params["scoring"])

#                 # 5) MLflow → metrics
#                 mlflow.log_metrics(metrics)

#                 # 6) MLflow → save model
#                 mlflow.sklearn.log_model(
#                     sk_model        = best_model,
#                     artifact_path   = "logistic_regression_model",
#                     registered_model_name = "BreastCancerLR"   
#                 )

#                 # 7) save the physical .pkl file (so the old workflow isn't disrupted).
#                 self.save_model(best_model)

#                 # 8) add the .pkl as an MLflow artifact
#                 mlflow.log_artifact(self.model_output_path)

#                 logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")

#             logger.info("Model training completed successfully")

#         except Exception as e:
#             logger.error(f"Error in model training pipeline: {e}")
#             raise CustomException("Failed during model training pipeline", e)


# # ──────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     trainer = ModelTraining(
#         PROCESSED_TRAIN_DATA_PATH,
#         PROCESSED_TEST_DATA_PATH,
#         MODEL_OUTPUT_PATH
#     )
#     trainer.run()






















import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = "Breast_Cancer_Logistic_Regression"


class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path, feature_store: RedisFeatureStore = None):
        self.train_path         = train_path
        self.test_path          = test_path
        self.model_output_path  = model_output_path
        self.feature_store      = feature_store       # Optional Redis feature store connection
        self.params_dist        = LOGISTIC_REGRESSION_PARAMS
        self.grid_search_params = GRID_SEARCH_PARAMS

        logger.info("Model Training initialized.")

    # ─────────────────────────────────────────────────────────────────
    def load_data_from_redis(self):
        """
        It retrieves data from the Redis Feature Store and splits it into train/test segments.
        """
        try:
            logger.info("Loading data from Redis...")

            entity_ids = self.feature_store.get_all_entity_ids()

            train_ids = [eid for eid in entity_ids if eid.startswith("train_")]
            test_ids  = [eid for eid in entity_ids if eid.startswith("test_")]

            train_data = [self.feature_store.get_features(eid) for eid in train_ids]
            test_data  = [self.feature_store.get_features(eid) for eid in test_ids]

            train_df = pd.DataFrame(train_data)
            test_df  = pd.DataFrame(test_data)

            logger.info(f"from Redis {len(train_df)} train, {len(test_df)} test records loaded.")
            return train_df, test_df

        except Exception as e:
            logger.error(f"Redis reading error: {e}")
            raise CustomException("Error loading data from Redis.", e)

    # ─────────────────────────────────────────────────────────────────
    def load_data_from_csv(self):
        """Classic reading from a CSV file — if you don't have Redis, this will work."""
        try:
            logger.info("Loading data from a CSV file....")
            train_df = load_data(self.train_path)
            test_df  = load_data(self.test_path)
            return train_df, test_df
        except Exception as e:
            logger.error(f"CSV reading error: {e}")
            raise CustomException("Error loading data from CSV.", e)

    # ─────────────────────────────────────────────────────────────────
    def split_features_labels(self, train_df, test_df):
        X_train = train_df.drop(columns=["diagnosis"])
        y_train = train_df["diagnosis"]
        X_test  = test_df.drop(columns=["diagnosis"])
        y_test  = test_df["diagnosis"]
        logger.info("completed")
        return X_train, y_train, X_test, y_test

    # ─────────────────────────────────────────────────────────────────
    def train_model(self, X_train, y_train):
        try:
            logger.info("GridSearchCV is starting...")

            lr_model = LogisticRegression(random_state=42)

            grid_search = GridSearchCV(
                estimator  = lr_model,
                param_grid = self.params_dist,
                cv         = self.grid_search_params["cv"],
                n_jobs     = self.grid_search_params["n_jobs"],
                verbose    = self.grid_search_params["verbose"],
                scoring    = self.grid_search_params["scoring"]
            )
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_, grid_search.best_params_

        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise CustomException("Error occurred while training the model.", e)

    # ─────────────────────────────────────────────────────────────────
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Model evaluation is starting...")

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy"  : accuracy_score(y_test, y_pred),
                "precision" : precision_score(y_test, y_pred),
                "recall"    : recall_score(y_test, y_pred),
                "f1"        : f1_score(y_test, y_pred),
                "roc_auc"   : roc_auc_score(y_test, y_proba),
            }

            for k, v in metrics.items():
                logger.info(f"{k}: {v:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            raise CustomException("Error occurred while evaluating the model.", e)

    # ─────────────────────────────────────────────────────────────────
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved: {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error occurred while saving the model: {e}")
            raise CustomException("Error occurred while saving the model.", e)

    # ─────────────────────────────────────────────────────────────────
    def run(self):
        try:
            logger.info("Model Training pipeline starting...")

            # 1) Upload data from Redis if you have Redis, otherwise from CSV.
            if self.feature_store:
                logger.info("Source: Redis Feature Store")
                train_df, test_df = self.load_data_from_redis()
            else:
                logger.info("Source: CSV files")
                train_df, test_df = self.load_data_from_csv()

            X_train, y_train, X_test, y_test = self.split_features_labels(train_df, test_df)

            # 2) MLflow tracking
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name="logistic_regression_gridsearch"):

                # Train
                best_model, best_params = self.train_model(X_train, y_train)

                # Evaluate
                metrics = self.evaluate_model(best_model, X_test, y_test)

                # MLflow → parameters
                mlflow.log_params(best_params)
                mlflow.log_param("cv_folds", self.grid_search_params["cv"])
                mlflow.log_param("scoring",  self.grid_search_params["scoring"])
                mlflow.log_param("data_source", "redis" if self.feature_store else "csv")

                # MLflow → metrics
                mlflow.log_metrics(metrics)

                # MLflow → model (sklearn flavor)
                mlflow.sklearn.log_model(
                    sk_model              = best_model,
                    artifact_path         = "logistic_regression_model",
                    registered_model_name = "BreastCancerLR"
                )

                self.save_model(best_model)
                mlflow.log_artifact(self.model_output_path)

                logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

            logger.info("Model Training pipeline completed.")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise CustomException("Error occurred in the model training pipeline.", e)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feature_store = RedisFeatureStore()   # Redis must be running in Docker.

    trainer = ModelTraining(
        train_path        = PROCESSED_TRAIN_DATA_PATH,
        test_path         = PROCESSED_TEST_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH,
        feature_store     = feature_store     # If you select None, it will read from the CSV.
    )
    trainer.run()