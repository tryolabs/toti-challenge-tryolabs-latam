import pandas as pd

from typing import Tuple, Union, List


class DataError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

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
        if target_column:
            if target_column not in data.columns:
                raise DataError(
                    message=f"target_column '{target_column}' not present in the raw data"
                )
            target = data[target_column]

        # Get the one-hot encoding of the columns.
        features = pd.DataFrame()
        for col, values in selected_features.items():
            for category in values:
                features[f"{col}_{category}"] = data[col] == category

        if target_column:
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
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
