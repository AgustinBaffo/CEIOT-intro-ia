import numpy as np
import pandas as pd


def generate_outliers(
    df: pd.DataFrame,
    columns: list = None,
    percentage: float = 0.01,
    extreme_outliers: bool = False,
    only_tails: bool = False,
    two_tailed: bool = True,
):
    """
    Generate outliers in the distribution of a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing features and one output.
    percentage (float): Percentage of total data points that should be replaced with outliers.
    extreme_outliers (bool): If True, it generates points with a big leverage.
    only_tails (bool): If True, generates outliers only in the tails (close to the min and max of the feature values).
    two_tailed (bool): If True, generates outliers from both parts of the feature (both in the min and the max).

    Returns:
    DataFrame: DataFrame with the newly generated outliers.
    """

    outlier_df = df.copy()

    if columns is None:
        columns = df.drop("target", axis=1).columns

    for column in columns:
        # Set the amount of leverage for the outliers
        leverage = 3 if extreme_outliers else 1.5

        # Calculate the IQR of the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Set the lower and upper bounds for outliers
        lower_bound = Q1 - (IQR * leverage)
        upper_bound = Q3 + (IQR * leverage)

        # Get the total number of outliers to be introduced
        num_outliers = int(df.shape[0] * percentage)

        # Depending on the settings, generate outliers at different places in the distribution
        if only_tails:
            if two_tailed:
                # Generate outliers in the lower and upper tails
                lower_indices = np.random.choice(
                    df[df[column] < (Q1 - (IQR))].index,
                    size=num_outliers // 2,
                    replace=True,
                )
                upper_indices = np.random.choice(
                    df[df[column] > (Q3 + (IQR))].index,
                    size=num_outliers // 2,
                    replace=True,
                )
                outlier_df.loc[lower_indices, column] = np.random.uniform(
                    lower_bound, Q1, size=num_outliers // 2
                )
                outlier_df.loc[upper_indices, column] = np.random.uniform(
                    Q3, upper_bound, size=num_outliers // 2
                )
            else:
                # Generate outliers in the upper tail
                upper_indices = np.random.choice(
                    df[df[column] > upper_bound].index, size=num_outliers, replace=True
                )
                outlier_df.loc[upper_indices, column] = np.random.uniform(
                    upper_bound, df[column].max(), size=num_outliers
                )
        else:
            # Generate outliers throughout the distribution
            indices = np.random.choice(
                df.index, size=num_outliers, replace=True)
            outlier_df.loc[indices, column] = np.random.uniform(
                df[column].min(), df[column].max(), size=num_outliers
            )

    return outlier_df


def data_split(X_input,
               Y_input,
               val_size=0.15,
               test_size=0.15,
               random_state=42,
               shuffle=True):

    _X_input = np.copy(X_input)
    _Y_input = np.copy(Y_input)

    if not _X_input.shape[0] == _Y_input.shape[0]:
        raise ValueError(
            "Los datos (X_input, Y_input) tienen distintas longitudes.")

    train_size = 1 - test_size - val_size

    if (train_size < 0):
        raise ValueError(
            "El porcentaje de datos de validacion y test no puede ser mayor al 100%.")

    # Mezclar los datos de manera tal que ambos conserven los mismos indices
    if (shuffle):
        np.random.seed(random_state)
        ran_idx = np.random.permutation(len(_X_input))
        _X_input = _X_input[ran_idx]
        _Y_input = _Y_input[ran_idx]

    total_len = _X_input.shape[0]
    train_len = int(train_size*total_len)
    val_len = int(val_size*total_len)

    X_train = np.array(_X_input[0:train_len])
    X_val = np.array(_X_input[train_len:train_len+val_len])
    X_test = np.array(_X_input[train_len+val_len:total_len])

    Y_train = np.array(_Y_input[0:train_len])
    Y_val = np.array(_Y_input[train_len:train_len+val_len])
    Y_test = np.array(_Y_input[train_len+val_len:total_len])

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
