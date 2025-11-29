import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin


class TabularDataProcessor:
    """Handles preprocessing and DataLoader creation for tabular data.

    Attributes:
        df: Source DataFrame
        feature_columns: Column names to use
        scaler: sklearn scaler for normalization
        device: PyTorch device
        X_train_raw: Raw training data after split
        X_val_raw: Raw validation data after split
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: List[str],
        scaler: Optional[TransformerMixin] = None,
        device: str = "cpu",
    ):
        """Initialize processor.

        Args:
            dataframe: Source data
            feature_columns: Columns to select
            scaler: sklearn scaler (defaults to StandardScaler)
            device: Device for tensors ('cpu' or 'cuda')

        Raises:
            ValueError: If feature_columns not found in dataframe
        """
        self.df = dataframe
        self.feature_columns = feature_columns
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.device = device

        self.X_train_raw: Optional[np.ndarray] = None
        self.X_val_raw: Optional[np.ndarray] = None

        missing = [col for col in feature_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    def process(
        self, batch_size: int = 32, val_split: float = 0.2, random_state: int = 42
    ) -> Tuple[DataLoader, int]:
        """Run full preprocessing pipeline.

        Steps: extract -> split -> fit scaler -> normalize -> create DataLoader

        Args:
            batch_size: Batch size for DataLoader
            val_split: Validation set fraction (0 to disable)
            random_state: Random seed for splitting

        Returns:
            Tuple of (train_loader, input_dim)
        """
        data_np = self.df[self.feature_columns].values.astype("float32")

        if val_split > 0:
            self.X_train_raw, self.X_val_raw = train_test_split(
                data_np, test_size=val_split, random_state=random_state
            )
        else:
            self.X_train_raw = data_np
            self.X_val_raw = None

        X_train_norm = self.scaler.fit_transform(self.X_train_raw)

        train_tensor = torch.from_numpy(X_train_norm).to(self.device)

        dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader, len(self.feature_columns)

    def get_scaler(self) -> TransformerMixin:
        """Return fitted scaler for inverse transformation."""
        return self.scaler
