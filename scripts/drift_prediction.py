import pandas as pd
import numpy as np
import shap.maskers
from sklearn.base import is_regressor, is_classifier
from sklearn.metrics import r2_score, accuracy_score


def predict_sklearn(
    model,
    explainer: shap.Explainer,
    previous_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_data: pd.Series,
    distance_metric="euclidean",
):
    def __euclidean_distance(X, Y):
        return np.sqrt(np.sum(np.square(X - Y)))

    def __manhattan_distance(X, Y):
        return np.sum(np.abs(X - Y))

    # Make prediction
    pred = model.predict(test_data)
    measure = 0
    if is_regressor(model):
        measure = r2_score(target_data, pred)
        print("MSE: ", measure)
    elif is_classifier(model):
        measure = accuracy_score(target_data, pred)
        print("Accuracy: ", measure)
    else:
        raise ValueError("Model is not a classifier or regressor.")

    # Explain previous data and test data
    prev_explain_values = explainer.shap_values(previous_data)
    test_explain_values = explainer.shap_values(test_data)

    prev_explain_means = np.mean(prev_explain_values, axis=0)
    test_explain_means = np.mean(test_explain_values, axis=0)

    # Normalize shap values
    prev_explain_means = (prev_explain_means - np.mean(prev_explain_means)) / np.std(
        prev_explain_means
    )
    test_explain_means = (test_explain_means - np.mean(test_explain_means)) / np.std(
        test_explain_means
    )

    # Calculate distance between previous and test data
    distances = []
    match distance_metric:
        case "euclidean":
            distances = [
                __euclidean_distance(prev_feat, test_feat)
                for prev_feat, test_feat in zip(prev_explain_means, test_explain_means)
            ]
        case "manhattan":
            distances = [
                __manhattan_distance(prev_feat, test_feat)
                for prev_feat, test_feat in zip(prev_explain_means, test_explain_means)
            ]
        case _:
            raise ValueError(
                "Invalid distance metric. Choose from 'euclidean' or 'manhattan'."
            )

    xai_results = pd.DataFrame(
        {
            "Previous explained importance": prev_explain_means,
            "Test explained importance": test_explain_means,
            "Distance": distances,
        },
        index=previous_data.columns,
    )

    return pred, xai_results, measure


def time_series_transform_df(
    data: pd.DataFrame,
    time_column: str,
    seasonality: tuple = (12, 'ME')# (n, d) where n is the number of periods and d is the period type (e.g. 'ME' for month end. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
) -> pd.DataFrame:
    
    interval, dim = seasonality

    if interval < 1 or interval >= len(data):
        raise ValueError("Seasonality interval out of bounds")
    
    season_df = data.copy().groupby(pd.Grouper(key=time_column, freq=dim.upper())).mean()

    columns = [f"{col}_t-{interval-i}" for col in season_df.columns for i in range(interval + 1)]
    result_df = pd.DataFrame(index=season_df.index[interval:], columns=columns)

    for col in season_df.columns:
        for i in range(interval + 1):
            result_df[f"{col}_t-{interval-i}"] = season_df[col].shift(interval - i)

    return result_df


def time_series_train(
    model,
    data,
    target,
    time_column,
    seasonality: tuple = (12 ,'ME')
):
    timeseries_data = time_series_transform_df(
        data=data, time_column=time_column, seasonality=seasonality
    )
    timeseries_target = target.iloc[seasonality[0]:].reset_index(drop=True)

    model.fit(timeseries_data, timeseries_target)

    return model, timeseries_data


