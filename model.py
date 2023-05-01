import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def preprocess(df: pd.DataFrame) -> ColumnTransformer:
    try:
        logging.info("Setting up data preprocessing module")
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
        logging.info(f"Numerical features to transform: {numeric_features}")
        categorical_features = df.select_dtypes(include=["object"]).columns
        logging.info(f"Categorical features to transform: {categorical_features}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        return preprocessor
    except Exception as e:
        logging.exception(e)


@dataclass
class DataSplit:
    X: pd.DataFrame
    y: pd.Series


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataSplit, DataSplit, DataSplit]:
    try:
        X = df.drop(columns=[target_col])
        logging.info(f"Encoding the target label {target_col}")
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df[target_col])
        y = label_encoder.transform(df[target_col])
        logging.info(
            f"Applying data split with the following ratio: {train_ratio}-{val_ratio}-{test_ratio}"
        )
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_ratio, random_state=seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_test,
            y_test,
            test_size=test_ratio / (test_ratio + val_ratio),
            random_state=seed,
        )
        train_set = DataSplit(x_train, y_train)
        val_set = DataSplit(x_val, y_val)
        test_set = DataSplit(x_test, y_test)
        logging.info("Successfully applied data split.")
        return train_set, val_set, test_set
    except Exception as e:
        logging.info(e)


def train(
    train_ds: DataSplit,
    val_ds: DataSplit,
    test_ds: DataSplit,
    model_data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> None:

    preprocessor = preprocess(train_ds.X)
    eval_set = [(preprocessor.fit_transform(val_ds.X), val_ds.y)]

    XGBPipe = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(**model_cfg)),
        ]
    )
    XGBPipe.fit(train_ds.X, train_ds.y, model__eval_set=eval_set)
    logging.info("validation score: %.3f" % XGBPipe.score(val_ds.X, val_ds.y))
    test_y_pred = XGBPipe.predict(test_ds.X)
    logging.info(classification_report(test_ds.y, test_y_pred))

    target_dir = model_data_cfg["target_dir"]
    if not os.path.exists(target_dir):
        logging.info(f"Creating directory {target_dir}")
        os.makedirs(target_dir)

    try:
        logging.info(f"Saving model to {target_dir}")
        model_file = model_data_cfg["model_file"]
        model_fullpath = os.path.join(target_dir, model_file)
        joblib.dump(XGBPipe, model_fullpath)
        logging.info(f"Model successfully saved to {model_fullpath}")
    except Exception as e:
        logging.exception(e)


def batch_predict(
    df: pd.DataFrame,
    target_col: str,
    model_data_cfg: Dict[str, Any],
    test_data_cfg: Dict[str, Any],
) -> None:
    logging.info("Starting model prediction")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    try:
        model_path = os.path.join(
            model_data_cfg["target_dir"], model_data_cfg["model_file"]
        )
        logging.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logging.exception(e)

    try:
        logging.info("Running predictions")
        _pred = model.predict(X)
        pred = label_encoder.inverse_transform(_pred)
    except Exception as e:
        logging.exception(e)

    result_df = pd.concat(
        [X.reset_index(drop=True), pd.DataFrame({'Adopted': y, 'Adopted_prediction': pred})], axis=1
    )
    target_dir = test_data_cfg["target_dir"]
    if not os.path.exists(target_dir):
        logging.info(f"Creating directory {target_dir}")
        os.makedirs(target_dir)
    output_file = test_data_cfg["output_file"]
    result_path = os.path.join(target_dir, output_file)
    logging.info(f"Writing predictions to {result_path}")
    result_df.to_csv(result_path, index=False)
    logging.info("Successfully written predictions. End of inferencing.")
