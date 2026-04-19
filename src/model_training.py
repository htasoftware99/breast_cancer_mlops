# import os
# import pandas as pd
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from config.model_params import *
# from utils.common_functions import load_data

# logger = get_logger(__name__)

# class ModelTraining:

#     def __init__(self, train_path, test_path, model_output_path):
#         self.train_path = train_path
#         self.test_path = test_path
#         self.model_output_path = model_output_path
#         self.params_dist = LOGISTIC_REGRESSION_PARAMS
#         self.grid_search_params = GRID_SEARCH_PARAMS

#     def load_and_split_data(self):
#         try:
#             logger.info(f"Loading train data from {self.train_path}")
#             train_df = load_data(self.train_path)

#             logger.info(f"Loading test data from {self.test_path}")
#             test_df = load_data(self.test_path)

#             X_train = train_df.drop(columns=["diagnosis"])
#             y_train = train_df["diagnosis"]

#             X_test = test_df.drop(columns=["diagnosis"])
#             y_test = test_df["diagnosis"]

#             logger.info("Data loaded and split successfully")
#             return X_train, y_train, X_test, y_test

#         except Exception as e:
#             logger.error(f"Error while loading data: {e}")
#             raise CustomException("Failed to load data", e)

#     def train_model(self, X_train, y_train):
#         try:
#             logger.info("Initializing Logistic Regression model")

#             lr_model = LogisticRegression(random_state=42)

#             logger.info("Starting hyperparameter tuning with GridSearchCV")

#             grid_search = GridSearchCV(
#                 estimator=lr_model,
#                 param_grid=self.params_dist,
#                 cv=self.grid_search_params["cv"],
#                 n_jobs=self.grid_search_params["n_jobs"],
#                 verbose=self.grid_search_params["verbose"],
#                 scoring=self.grid_search_params["scoring"]
#             )

#             grid_search.fit(X_train, y_train)

#             best_params = grid_search.best_params_
#             best_model = grid_search.best_estimator_

#             logger.info(f"Best params: {best_params}")
#             return best_model

#         except Exception as e:
#             logger.error(f"Error while training model: {e}")
#             raise CustomException("Failed to train model", e)

#     def evaluate_model(self, model, X_test, y_test):
#         try:
#             logger.info("Evaluating model")

#             y_pred  = model.predict(X_test)
#             y_proba = model.predict_proba(X_test)[:, 1]

#             accuracy  = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall    = recall_score(y_test, y_pred)
#             f1        = f1_score(y_test, y_pred)
#             roc_auc   = roc_auc_score(y_test, y_proba)

#             logger.info(f"Accuracy  : {accuracy}")
#             logger.info(f"Precision : {precision}")
#             logger.info(f"Recall    : {recall}")
#             logger.info(f"F1 Score  : {f1}")
#             logger.info(f"ROC-AUC   : {roc_auc}")

#             return {
#                 "accuracy"  : accuracy,
#                 "precision" : precision,
#                 "recall"    : recall,
#                 "f1"        : f1,
#                 "roc_auc"   : roc_auc
#             }

#         except Exception as e:
#             logger.error(f"Error while evaluating model: {e}")
#             raise CustomException("Failed to evaluate model", e)

#     def save_model(self, model):
#         try:
#             os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
#             joblib.dump(model, self.model_output_path)
#             logger.info(f"Model saved to {self.model_output_path}")

#         except Exception as e:
#             logger.error(f"Error while saving model: {e}")
#             raise CustomException("Failed to save model", e)

#     def run(self):
#         try:
#             logger.info("Starting Model Training pipeline")

#             X_train, y_train, X_test, y_test = self.load_and_split_data()
#             best_model = self.train_model(X_train, y_train)
#             metrics    = self.evaluate_model(best_model, X_test, y_test)
#             self.save_model(best_model)

#             logger.info("Model training completed successfully")

#         except Exception as e:
#             logger.error(f"Error in model training pipeline: {e}")
#             raise CustomException("Failed during model training pipeline", e)


# if __name__ == "__main__":
#     trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
#     trainer.run()
















import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data

logger = get_logger(__name__)

# ─── MLflow settings ───────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "Breast_Cancer_Logistic_Regression"
# for local UI: mlflow ui  →  http://127.0.0.1:5000
# DagsHub için bu satırı açıp tracking URI'yi değiştir:
# mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")
# ───────────────────────────────────────────────────────────────────


class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path       = train_path
        self.test_path        = test_path
        self.model_output_path = model_output_path
        self.params_dist      = LOGISTIC_REGRESSION_PARAMS
        self.grid_search_params = GRID_SEARCH_PARAMS

    # ──────────────────────────────────────────────────────────────
    def load_and_split_data(self):
        try:
            logger.info(f"Loading train data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading test data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["diagnosis"])
            y_train = train_df["diagnosis"]

            X_test  = test_df.drop(columns=["diagnosis"])
            y_test  = test_df["diagnosis"]

            logger.info("Data loaded and split successfully")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)

    # ──────────────────────────────────────────────────────────────
    def train_model(self, X_train, y_train):
        try:
            logger.info("Initializing Logistic Regression model")

            lr_model = LogisticRegression(random_state=42)

            logger.info("Starting hyperparameter tuning with GridSearchCV")
            grid_search = GridSearchCV(
                estimator   = lr_model,
                param_grid  = self.params_dist,
                cv          = self.grid_search_params["cv"],
                n_jobs      = self.grid_search_params["n_jobs"],
                verbose     = self.grid_search_params["verbose"],
                scoring     = self.grid_search_params["scoring"]
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_model  = grid_search.best_estimator_

            logger.info(f"Best params: {best_params}")
            return best_model, best_params

        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model", e)

    # ──────────────────────────────────────────────────────────────
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")

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
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    # ──────────────────────────────────────────────────────────────
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)

    # ──────────────────────────────────────────────────────────────
    def run(self):
        try:
            logger.info("Starting Model Training pipeline")

            # ── MLflow Experiment ──────────────────────────────────
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name="logistic_regression_gridsearch"):

                # 1) Veriyi yükle
                X_train, y_train, X_test, y_test = self.load_and_split_data()

                # 2) Modeli eğit
                best_model, best_params = self.train_model(X_train, y_train)

                # 3) Değerlendir
                metrics = self.evaluate_model(best_model, X_test, y_test)

                # 4) MLflow → parametreler
                mlflow.log_params(best_params)

                # GridSearch config'ini de logla
                mlflow.log_param("cv_folds",  self.grid_search_params["cv"])
                mlflow.log_param("scoring",   self.grid_search_params["scoring"])

                # 5) MLflow → metrikler
                mlflow.log_metrics(metrics)

                # 6) MLflow → modeli kaydet (sklearn flavor)
                mlflow.sklearn.log_model(
                    sk_model        = best_model,
                    artifact_path   = "logistic_regression_model",
                    registered_model_name = "BreastCancerLR"   # Model Registry'ye de kaydeder
                )

                # 7) Fiziksel .pkl dosyasını da kaydet (eski akış bozulmasın)
                self.save_model(best_model)

                # 8) .pkl'yi MLflow artifact'ı olarak da ekle
                mlflow.log_artifact(self.model_output_path)

                logger.info(f"MLflow run tamamlandı. Run ID: {mlflow.active_run().info.run_id}")

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e)


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH,
        PROCESSED_TEST_DATA_PATH,
        MODEL_OUTPUT_PATH
    )
    trainer.run()