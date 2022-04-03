"""

Classes and methods that turn a humble regressor into a mighty forecaster

"""
import abc
import copy
import logging
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

import numpy as np
import pandas as pd


class MultipleTimestepRegressionForecaster(abc.ABC):
    """Wraps regression objects to forecast multiple timepoints ahead of predictor features. Contains multiple instances
    of the SingleTimestepRegressionForecaster class.

    Note: the recursive forecasting options are experimental and not currently recommended.
    """

    def __init__(
        self,
        regressor,
        horizon,
        forecast_type="proportional_diff",
        meta_feature_prefix=None,
        recursion_strat=None,
        string_results_format="%Y-%m-%d",
    ):
        """
        Args:
            regressor (scikit-learn-like regression instance or list of instances): The regressor object or objects to be used as forecasters.
            horizon ([int or list of ints]): If int, the number of timesteps into the future to forecast, including all previous timesteps. If a list, the specific timesteps to be forecast.
            recursion_strat (str, optional): If set to "recursive-single", the value forecast at each timestep is given the the forecaster for the next timestep as an input feature. If set to "recursive-cumulative", the values forecasted at all prior timesteps are input as input features. Defaults to None (no recursion).
            string_results_format (str, optional): If not set to None, the date format of the strings which index the dictionary of forecasts, in the case of multiple forecast start dates. Defaults to "%Y-%m-%d".

        Raises:
            ValueError: The number of timesteps must match the length of the string of regressor instances, if a list of regressors is input rather than a single instance.
        """
        # Horizon can be an interable of timesteps, or an integer
        if isinstance(horizon, int):
            self.timesteps = range(1, horizon + 1)
        elif isinstance(horizon, list):
            self.timesteps = horizon
        # Create the list of forecasters.
        self.forecasters = []
        if isinstance(regressor, list) is False:
            for timestep in self.timesteps:
                r = copy.deepcopy(regressor)
                # Maybe copy.deepcopy() isn't the right way to copy some regressors, be careful
                self.forecasters.append(
                    SingleTimestepRegressionForecaster(
                        copy.deepcopy(r),
                        timestep,
                        forecast_type=forecast_type,
                        meta_feature_prefix=meta_feature_prefix,
                    )
                )
        else:
            if len(regressor) != len(self.timesteps):
                raise ValueError(
                    "If a list of regressors is given then its length must match the number of timesteps."
                )
            for i, t in enumerate(self.timesteps):
                r = regressor[i]
                self.forecasters.append(
                    SingleTimestepRegressionForecaster(
                        r,
                        t,
                        forecast_type=forecast_type,
                        meta_feature_prefix=meta_feature_prefix,
                    )
                )
        self._string_results_format = string_results_format
        self._horizon = horizon
        self.forecast_type = forecast_type
        self.recursion_strat = recursion_strat

    def __repr__(self):
        reprstring = (
            f"MultipleTimestepRegressionForecaster(horizon={self._horizon}, "
            f"forecast_type={self.forecast_type}, "
            f"recursion_strat={self.recursion_strat}, forecasters={self.forecasters})"
        )
        return reprstring

    def fit(self, X: pd.DataFrame, y, freq=None, meta_features=None):
        """Fit the forecasters to the predictors by calling either the recursive or non-recursive fit method.

        Args:
            X (pd.DataFrame of shape (n_timepoints, n_features)): Predictor dataframe.
            y (string): The name of the column to be forecast.
            freq (pandas.tseries.offsets.MonthEnd, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if self.recursion_strat is None:
            return self._fit_nonrecursive(X, y, freq, meta_features=meta_features)
        elif self.recursion_strat in ("recursive-single", "recursive-cumulative"):
            return self._fit_recursive(X, y, freq)

    def _fit_nonrecursive(self, X, target, freq, meta_features=None):
        # Sequentially fit forecasters
        for i, forecaster in enumerate(self.forecasters):
            self.forecasters[i] = forecaster.fit(
                X, target, freq=freq, meta_features=meta_features
            )
        return self

    def _fit_recursive(self, X, target, freq):
        # Sequentially fit forecasters with forecasts from previous forecasters used as input features.
        recursive_features = pd.DataFrame(index=X.index)
        for i, forecaster in enumerate(self.forecasters):
            X_full = pd.concat((X, recursive_features), axis=1)
            self.forecasters[i] = forecaster.fit(X_full, target, freq=freq)
            forecast = self.forecasters[i].predict(X_full)
            recursive_column_names = [
                f"target_forecast_y{x}_t{i + 1}" for x in range(forecast.shape[1])
            ]
            recursive_feature_last = pd.DataFrame(
                data=forecast.to_numpy(), index=X.index, columns=recursive_column_names
            )
            if self.recursion_strat == "recursive-single":
                recursive_features = recursive_feature_last
            elif self.recursion_strat == "recursive-cumulative":
                recursive_features = pd.concat(
                    (recursive_features, recursive_feature_last), axis=1
                )
        return self

    @staticmethod
    def _predict_single(self, X=None, meta_features=None, conf_ints=False):
        """Predicts a single set of forecasts starting from a single row of predictors.

        Args:
            X (pandas.DataFrame of shape (1, n_features), optional): A single row of input features. Defaults to None.

        Returns:
            pandas.DataFrame of shape (n_timepoints, n_forecasted_values): The forecasted y values.
        """
        if (
            X is None or self.recursion_strat is None
        ):  # If no X is given as an argument the last set of inputs from the training data
            # will be used, which already include the recursive features, therefore the nonrecursive prediction method is called.
            forecasts = self._predict_single_nonrecursive(
                self, X=X, meta_features=meta_features, conf_ints=conf_ints
            )
        else:
            forecasts = self._predict_single_recursive(self, X=X, conf_ints=conf_ints)

        if len(forecasts) > 1:
            return pd.concat(forecasts, axis=0)
        else:
            return forecasts[0]

    @staticmethod
    def _predict_single_nonrecursive(self, X=None, meta_features=None, conf_ints=False):
        """Loop through the list of forecasters and returns the forecasts in a list."""
        forecasts = []
        for forecaster in self.forecasters:
            forecast = forecaster.predict(
                X, meta_features=meta_features, conf_ints=conf_ints
            )
            forecasts.append(forecast)
        return forecasts

    @staticmethod
    def _predict_single_recursive(self, X=None, meta_features=None, conf_ints=False):
        """Loop through the list of forecasters and returns the forecasts in a list."""
        forecasts = []
        recursive_features = pd.DataFrame(index=X.index)
        for i, forecaster in enumerate(self.forecasters):
            X_full = pd.concat((X, recursive_features), axis=1)
            forecast = forecaster.predict(
                X_full, meta_features=meta_features, conf_ints=conf_ints
            )
            forecasts.append(forecast)
            recursive_column_names = [
                f"target_forecast_y{x}_t{i + 1}" for x in range(forecast.shape[1])
            ]
            recursive_feature_last = pd.DataFrame(
                data=forecast.to_numpy(), index=X.index, columns=recursive_column_names
            )
            if self.recursion_strat == "recursive-single":
                recursive_features = recursive_feature_last
            elif self.recursion_strat == "recursive-cumulative":
                recursive_features = pd.concat(
                    (recursive_features, recursive_feature_last), axis=1
                )
        return forecasts

    def predict(self, X=None, meta_features=None, conf_ints=False):
        """Predicts set or sets of forecasts.

        Args:
            X (pandas.DataFrame of shape (n_timepoints, n_features), optional): Row or rows of predictor features. The data from last date in the training set is used if not input is provided. Defaults to None.

        Returns:
            [pandas.DataFrame or dict of DataFrames]: If X has multiple rows, results will be returned as DataFrames in a dictionary indexed by the forecast start point (i.e. the timepoint of the input).
        """

        if X is None or X.shape[0] == 1:
            forecast = self._predict_single(
                self, X=X, meta_features=meta_features, conf_ints=conf_ints
            )
            return forecast
        else:
            results_dict = {}
            for idx in X.index:
                X_single = pd.DataFrame(X.loc[idx, :]).T
                if self._string_results_format:
                    dict_idx = idx.strftime(self._string_results_format)
                else:
                    dict_idx = idx
                results_dict[dict_idx] = self._predict_single(
                    self, X_single, meta_features=meta_features, conf_ints=conf_ints
                )
            return results_dict


class SingleTimestepRegressionForecaster(abc.ABC):
    """Wrapper for a scikit-learn-like regression object that forecasts a single future timepoint."""

    def __init__(
        self,
        regressor,
        timestep,
        forecast_type="proportional_diff",
        meta_feature_prefix=None,
    ):
        # forecast_type: "proportional_diff", "absolute_diff", "absolute"
        self.regressor = regressor
        if timestep <= 0:
            raise ValueError("Timestep argument must be greater than 0")
        self.timestep = timestep
        self.forecast_type = forecast_type
        self.meta_feature_prefix = meta_feature_prefix
        if forecast_type == "absolute" and meta_feature_prefix is not None:
            raise ValueError(
                "Can't have both absolute value prediction and meta feature boosting!"
            )

    def __repr__(self):
        reprstring = (
            f"SingleTimestepRegressionForecaster(timestep={self.timestep}, regressor={self.regressor},"
            f" forecast_type={self.forecast_type})"
        )
        return reprstring

    def fit(self, X: pd.DataFrame, y, freq=None, meta_features=None, **kwargs):
        """Fit the enclosed regressor to forecast the value of y *timestep* periods in the future.

        Args:
            X (pd.DataFrame of shape (n_timepoints, n_features)): Dataframe of predictor values.
            target (string): name of the y column to be forecast.
            freq (pandas.teseries.offset, optional): The frequency of the y series, to be provided if the freq attribute of the index of the y DataFrame has not been set. Defaults to None.

        Raises:
            ValueError: If frequency of the series is not provided as either an argument or an attribute of the y.

        Returns:
            pandas.DataFrame: DataFrame of forecasted values.
        """
        # Find the freq of the timeseries and store it
        if y.index.freq is None:
            if freq is None:
                raise ValueError(
                    "Frequency must be provided as an argument if not in index of y"
                )
            else:
                self.freq = freq
        else:
            if freq is not None and freq != y.index.freq:
                logging.error(
                    "Value of freq argument does not match freq of y, using freq of y instead"
                )
            self.freq = y.index.freq

        # Align X and y
        X, y = X.align(y, join="inner", axis=0)

        # Store the last entries in X and y to enable forecasting without input
        self.last_X = pd.DataFrame(X.loc[X.index.max(), :]).T
        self.last_y = pd.DataFrame(y.loc[X.index.max(), :]).T

        shifted_y = y.shift(
            -self.timestep
        ).dropna()  # Dropna because shifting creates missing values
        if self.meta_feature_prefix is not None:
            # Get the meta feature for the forecaster's timestep
            self.last_meta_feature = pd.DataFrame(meta_features.loc[X.index.max(), :]).T
            meta_feature_mean = meta_features[
                self.meta_feature_prefix + f"_mean_t{self.timestep}"
            ]
            # Align the meta feature with the y
            shifted_y, meta_feature_mean = shifted_y.align(
                meta_feature_mean, join="inner", axis=0
            )
            if self.forecast_type == "absolute_diff":
                shifted_y = shifted_y - meta_feature_mean.to_numpy().reshape(
                    shifted_y.shape
                )
            if self.forecast_type == "proportional_diff":
                shifted_y = shifted_y / meta_feature_mean.to_numpy().reshape(
                    shifted_y.shape
                )
        else:
            if self.forecast_type == "absolute_diff":
                shifted_y = shifted_y - y.to_numpy()
            elif self.forecast_type == "proportional_diff":
                shifted_y = shifted_y / y.to_numpy()
        shifted_y, X = shifted_y.align(
            X, join="inner", axis=0
        )  # Reindex X to drop rows dropped in shifted_y
        # X, shifted_y = reduce_mem_usage(X), reduce_mem_usage(shifted_y)
        self.regressor = self.regressor.fit(X, shifted_y.to_numpy().ravel(), **kwargs)
        return self

    @staticmethod
    def _get_forecast_index(self, X):
        """Utility method to calculate the correct indexes of the forecasted values.

        Args:
            X (pandas.DataFrame): The DataFrame of predictor values. The returned indexes are self.timestep points in the future.

        Returns:
            pandas.DateTimeIndex: Indexes for the forecasted values.
        """
        forecast_index = X.index + self.freq * self.timestep
        return forecast_index

    def predict(self, X=None, y=None, meta_features=None, conf_ints=False):
        """Forecast y values with the enclosed regressor.

        Args:
            X (pandas.DataFrame of shape (n_timepoints, n_features), optional): DataFrame of predictors. The last row of predictors in the training set will be used if none is provided. Defaults to None.

        Returns:
            [pandas.DataFrame of shape (n_timepoints, 1)]: Forecasted values.
        """
        if y is not None and X is not None:
            X, y = X.align(y, join="inner", axis=0)
        elif X is None:
            X = self.last_X
            y = self.last_y
            if self.meta_feature_prefix is not None:
                meta_features = self.last_meta_feature

        forecast = self.regressor.predict(X)
        if self.meta_feature_prefix is not None:
            X, meta_features = X.align(meta_features, join="left", axis=0)
            meta_feature_mean = meta_features[
                self.meta_feature_prefix + f"_mean_t{self.timestep}"
            ]
            if self.forecast_type == "proportional_diff":
                forecast = forecast * meta_feature_mean.to_numpy().ravel()
            elif self.forecast_type == "absolute_diff":
                forecast = forecast + meta_feature_mean.to_numpy().ravel()
        elif self.forecast_type == "absolute_diff":
            forecast = forecast + y.to_numpy().ravel()
        elif self.forecast_type == "proportional_diff":
            forecast = forecast * y.to_numpy().ravel()
        if conf_ints is True:
            forecast = self._get_conf_ints(self, forecast, meta_features)
            forecast_columns = ("forecast", "lower", "upper")
        else:
            forecast_columns = ("forecast",)
        forecast_index = self._get_forecast_index(self, X)
        forecast = pd.DataFrame(
            data=forecast, index=forecast_index, columns=forecast_columns
        )
        return forecast

    @staticmethod
    def _get_conf_ints(self, mean_forecast=None, meta_features=None):
        if meta_features is None:
            raise ValueError("meta_features is needed for confidence intervals")
        mean_forecast = mean_forecast.reshape((-1, 1))
        meta_feature_cols = [
            self.meta_feature_prefix + s + f"_t{self.timestep}"
            for s in ("_mean", "_lower", "_upper")
        ]
        # meta_features_current = meta_features[meta_feature_cols]
        cis_relative = meta_features[meta_feature_cols[1:]].to_numpy().reshape(
            (-1, 2)
        ) - meta_features[meta_feature_cols[0]].to_numpy().reshape((-1, 1))
        cis_recentred = cis_relative + mean_forecast
        forecast = np.concatenate((mean_forecast, cis_recentred), axis=1)
        return forecast


class WrappedQuantileRegressor(abc.ABC):
    """
    This is an aggregation of scikit-learn-like regression objects that are capable of quantile regression, that
    simultaneously gives predictions for specified quantiles. It is intended to be used with the
    SingleTimestepRegressionForecaster class.
    """

    def __init__(
        self,
        regressor,
        quantiles,
        quantile_kw="alpha",
        objective_kwarg="quantile",
        **regressor_kwargs,
    ):
        """[summary]

        Args:
            regressor (scikit-learn-like regressor class): Scikit-learn-like regression class capable of quantile regression.
            quantiles (list-like): Container of quantile values to be estimated.
            quantile_kw (str, optional): String definition of the keyword argument of the passed regression class which sets the quantile to be estimated. Defaults to "alpha" (keyword for the LightGBM.LGBMRegressor class).
            objective_kwarg (str, optional): The value of the "objective" kwarg required to set the regressor class to perform quantile regression. Defaults to "quantile" (the value for the LightGBM.LGBMRegressor class).
        """
        self.quantiles = quantiles
        self.regressor_kwargs = regressor_kwargs
        self.regressors = []
        for quantile in quantiles:
            quantile_kwargs = {quantile_kw: quantile}
            if objective_kwarg is not None:
                quantile_kwargs["objective"] = objective_kwarg
            # The objective argument might be called something else in some models
            self.regressors.append(regressor(**quantile_kwargs, **regressor_kwargs))
            # self.regressors.append(regressor(alpha=quantiles[i], objective="quantile", **regressor_kwargs))

    def __repr__(self):
        reprstring = f"WrappedQuantileRegressor(quantiles={self.quantiles}, regressors={self.regressors})"
        return reprstring

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fits the wrapped regressors.

        Args:
            X (array-like of shape = [n_samples x n_features]): container of predictors
            y (array-like of shape = [n_samples x 1]): container of y values
            **kwargs: Other parameters passed to the underlying wrapped regressors.

        Returns:
            A fitted WrappedQuantileRegressor.

        """
        for i in range(len(self.quantiles)):
            self.regressors[i].fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Calls predict on the wrapped regressors and returns the results in a list.

        Args:
            X (array-like): n_sample x n_features container of predictors.
            **kwargs: Other parameters passed to the underlying wrapped regressors.

        Returns:
            A list of predicted quantile values.

        """
        y_predictions = []
        for i in range(len(self.quantiles)):
            y_predictions.append(self.regressors[i].predict(X, **kwargs))
        # Turn the list of predictions into a 2d array
        y_predictions = [np.reshape(y, (-1, 1)) for y in y_predictions]
        y_predictions = np.concatenate(y_predictions, axis=1)
        return y_predictions
