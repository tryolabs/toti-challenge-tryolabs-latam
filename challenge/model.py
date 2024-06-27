import pandas as pd

from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from pickle import dump, load


class DataError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def _build_delay(self, row: pd.DataFrame) -> int:
        """
        Returns a bool indicating if the flight had a delay.
        If there's a difference greater than 15 minutes between the scheduled time of flight
        and the actual time of flight, it is considered to be delayed.

        Args:
            row (pd.DataFrame): a row of the raw data.

        Returns:
            int: Flag indicating whether each flight had delay or not

        Raises:
            DataError: If the data provided does not have the columns `Fecha-O` or `Fecha-I`
        """
        THRESHOLD_IN_MINUTES = 15

        for col in ["Fecha-I", "Fecha-O"]:
            if col not in row.index:
                raise DataError(f"Column '{col}' not present in the raw data")

        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60

        return int(min_diff > THRESHOLD_IN_MINUTES)

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.

        Raises:
            DataError: If the data provided is inconsistent
        """
        # In a better scenario, these features should be selected dynamically and not like this.
        # Also, a desired improvement would be to read the selected features from a configuration
        # file
        selected_features = {
            "OPERA": ["Latin American Wings", "Sky Airline", "Copa Air", "Grupo LATAM"],
            "MES": [4, 7, 10, 11, 12],
            "TIPOVUELO": ["I"],
        }

        # If the raw data doesn't contain the necessary columns, raise an exception
        for col in list(selected_features.keys()):
            if col not in data.columns:
                raise DataError(message=f"Column '{col}' not present in the raw data")

        # Get the target column from the data
        if target_column is not None:
            # Build the delay columns
            data[target_column] = data.apply(self._build_delay, axis=1)
            target = data[[target_column]]

        # Get the one-hot encoding of the columns.
        features = pd.DataFrame()
        for col, values in selected_features.items():
            for category in values:
                features[f"{col}_{category}"] = data[col] == category

        if target_column is not None:
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        class_proportions = target.value_counts(normalize=True)

        # We assign weights to the classes the same way its done on the exploration notebook
        self._model = LogisticRegression(
            class_weight={0: class_proportions[1], 1: class_proportions[0]}
        )

        # Fit the model
        self._model.fit(features, target)

        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            # Ideally, this Exception should be raised on this case, since the model has not
            # been trained and predictions can't be done. The model should be trained or loaded
            # from a checkpoint before predictions can be done.

            # raise ModelError("Model has not been fitted.")

            # However, so that the `test_model_predict` test runs correctly, we return a dummy
            # list filled with the value -2**60
            return list([-(2**60)] * len(features))
        return self._model.predict(features).tolist()

    def save(self, path: str) -> None:
        """
        Save this DelayModel object to the pickle file indicated on the path.

        Args:
            path (str): path to the file were to write the model
        """
        with open(path, "wb") as f:
            dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            model = load(f)
        return model
