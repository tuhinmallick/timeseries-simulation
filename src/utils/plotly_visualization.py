import pandas as pd
import numpy as np

import plotly.graph_objects as go


def plotly_plot_forecast(
    df_actuals, df_forecast, display_fig=True, figsize=(1400, 500), **kwargs
):
    """Function to generate the forecast plot with actuals and confidence intervall.

    Args:
        df_actuals (pandas.DataFrame): Actual values of the target series with date on index. Need column name "actual".
        df_forecast (pandas.DataFrame): Forecasted values and confidence intervalls of the target series with date on index. Need column name "forecast", "lower", "upper".
        display_fig (bool, optional): Select if plot is displayed or figure object is returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the plotly figure object if display_fig is True.
    """

    df_test = pd.concat([df_actuals, df_forecast], axis=1)

    fig = go.Figure()

    # ================================ ACTUALS AND FORECAST CONNECTION W/ CI ================================
    df_subline = df_forecast.dropna()
    df_subline.columns = ["actual", "lower", "upper"]
    df_subline = df_actuals.iloc[-1:].append(df_subline.dropna().iloc[:1])

    for i, _ in enumerate(df_subline.columns):
        df_subline.iloc[0, i] = df_actuals.iloc[-1:, 0]

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_subline.index, df_subline.index[::-1]]),
            y=np.concatenate([df_subline["upper"], df_subline["lower"][::-1]]),
            marker_color="rgba(98,249,252,0.0)",
            fill="toself",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_subline.index,
            y=round(df_subline["actual"], 1),
            marker_color="rgba(98,249,252,0.9)",
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ================================ FORECAST, ACTUALS AND CI ================================
    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["actual"], 1),
            name="Actuals",
            marker_color="rgba(242,242,242,1)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_test.index, df_test.index[::-1]]),
            y=np.concatenate([df_test["upper"], df_test["lower"][::-1]]),
            name="CI*",
            fill="toself",
            hoverinfo="skip",
            marker_color="rgba(98,249,252,0.0)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["forecast"], 1),
            name="Forecast",
            marker_color="rgba(98,249,252,0.9)",
        )
    )

    # ================================ PLOT AESTHETICS ================================
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
    )

    if display_fig == True:
        fig.show()
    else:
        return fig


def plotly_plot_backtest(
    df_backtest, df_forecast=None, display_fig=True, figsize=(1400, 500), **kwargs
):
    """Function to generate the backtest plot with actuals, forecast and confidence intervall.

    Args:
        df_backtest (pandas.DataFrame): Backtesting results with date on the index.
        df_forecast (pandas.DataFrame): Forecasted values and confidence intervalls of the target series with date on index. Need column name "forecast", "lower", "upper".
        display_fig (bool, optional): Select if plot is displayed or figure object is returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the plotly figure object if display_fig is True.
    """

    fig = go.Figure()

    # ================================ INCLUDE THE FORECAST ================================
    if isinstance(df_forecast, pd.DataFrame):
        df_forecast1 = df_forecast.copy()
        df_forecast1.columns = ["h_forecast", "h_lower", "h_upper"]
        df_test = pd.concat([df_backtest, df_forecast1.dropna()], axis=1)
        # Fix the latest value so the lines are connected.

        # ================================ ACTUALS AND FORECAST CONNECTION W/ CI ================================
        df_subline = df_forecast.dropna()
        df_subline.columns = ["actual", "lower", "upper"]
        df_subline = (
            df_backtest.drop(["forecast"], axis=1).iloc[-1:].append(df_subline.iloc[:1])
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_subline.index, df_subline.index[::-1]]),
                y=np.concatenate([df_subline["upper"], df_subline["lower"][::-1]]),
                marker_color="rgba(98,249,252,0.0)",
                fill="toself",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # fig.add_trace(
        #     go.Scatter(
        #         x=df_subline.index,
        #         y=round(df_subline["actual"], 1),
        #         marker_color="rgba(98,249,252,0.9)",
        #         mode="lines",
        #         showlegend=False,
        #         hoverinfo="skip"
        #         )
        #     )

        # NOTE Add the line connection between forecast and historical forecast:
        df_subline = df_forecast1.dropna()
        df_subline.columns = ["forecast", "lower", "upper"]
        df_subline = (
            df_backtest.drop(["actual"], axis=1).iloc[-1:].append(df_subline.iloc[:1])
        )

        # ================================ HISTORICAL FORECAST AND FORECAST LINE CONNECTION ================================
        fig.add_trace(
            go.Scatter(
                x=df_subline.index,
                y=round(df_subline["forecast"], 1),
                marker_color="rgba(98,249,252,0.9)",
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # ================================ FORECAST W/ CONFIDENCE INTERVAL ================================
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_test.index, df_test.index[::-1]]),
                y=np.concatenate([df_test["h_upper"], df_test["h_lower"][::-1]]),
                marker_color="rgba(98,249,252,0.0)",
                name="CI*",
                fill="toself",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=round(df_test["h_forecast"], 1),
                name="Forecast",
                marker_color="rgba(98,249,252,0.9)",
            )
        )

        df_test = df_backtest.copy()
    else:
        df_test = df_backtest.copy()

    # ================================ HISTORICAL FORECAST, ACTUALS AND CONFIDENCE INTERVALL ================================
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_test.index, df_test.index[::-1]]),  #
            y=np.concatenate([df_test["upper"], df_test["lower"][::-1]]),
            marker_color="rgba(141,201,40,0.0)",
            name="Historical CI*",
            fill="toself",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["forecast"], 1),
            name="Historical Forecast",
            marker_color="rgba(141,201,40,0.9)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["actual"], 1),
            name="Actuals",
            line=dict(width=3),
            marker_color="rgba(242,242,242,1)",
        )
    )

    # ================================ PLOT AESTHETICS ================================
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
    )

    if display_fig == True:
        fig.show()
    else:
        return fig


def plotly_feature_importance(
    df_features, number_features=10, display_fig=True, figsize=(1400, 300), **kwargs
):
    """Function to plot the feature importance of the model.

    Args:
        df_features (pandas.DataFrame): Feature importances per feature in %.
        number_features (int, optional): Number of features to be displayed in the bar chart. Defaults to 10.
        display_fig (bool, optional): Select if the figure is displayed or returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the bar chart with the feature importances and the features if display_fig equals True.
    """

    fig = go.Figure()

    col_names = df_features.columns.tolist()

    # ================================ DRIVERS PLOT ================================
    fig.add_trace(
        go.Bar(
            y=df_features.loc[:number_features, col_names[0]],
            x=df_features.loc[:number_features, col_names[1]],
            orientation="h",
            marker_color="rgba(98,249,252,0.9)",
        )
    )

    # ================================ PLOT AESTHETICS ================================
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="Feature Importance (%)",
        # yaxis_title=f"Price, USD/toz",
        hovermode="y",
        template="plotly_dark",
        margin=dict(l=80, r=30, t=30, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
    )

    if display_fig == True:
        fig.show()
    else:
        return fig


def plotly_plot_simulation(
    df_actuals,
    df_forecast,
    df_simulation,
    conf_intervall=True,
    display_fig=True,
    figsize=(1400, 500),
    **kwargs,
):
    """Function to generate the forecast plot with actuals and confidence intervall.

    Args:
        df_actuals (pandas.DataFrame): Actual values of the target series with date on index. Need column name "actual".
        df_forecast (pandas.DataFrame): Forecasted values and confidence intervalls of the target series with date on index. Need column name "forecast", "lower", "upper".
        df_simulation (pandas.DataFrame): Simulated values and confidence intervalls of the target series with date on index. Need column name "forecast", "lower", "upper".
        horizon (int, optional): Horizon value to fix adjust the combined dataframe and to make line consistent. Defaults to 6.
        conf_interval (bool, optional): Select if the confidence intervalls are plotted too. Defaults to True.
        display_fig (bool, optional): Select if plot is displayed or figure object is returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the plotly figure object if display_fig is True.
    """

    df_test = pd.concat([df_actuals, df_forecast], axis=1)
    df_test_simulation = pd.concat([df_actuals, df_simulation], axis=1)

    fig = go.Figure()

    # ================================ ACTUALS AND FORECAST CONNECTION W/ CI ================================
    df_subline1 = df_forecast.dropna()
    df_subline1.columns = ["actual", "lower", "upper"]
    df_subline1 = df_actuals.iloc[-1:].append(df_subline1.dropna().iloc[:1])

    for i, _ in enumerate(df_subline1.columns):
        df_subline1.iloc[0, i] = df_actuals.iloc[-1:, 0]

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_subline1.index, df_subline1.index[::-1]]),
            y=np.concatenate([df_subline1["upper"], df_subline1["lower"][::-1]]),
            marker_color="rgba(98,249,252,0.0)",
            fill="toself",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ================================ ACTUALS AND SIMULATION CONNECTION W/ CI ================================
    df_subline = df_simulation.dropna()
    df_subline.columns = ["actual", "lower", "upper"]
    df_subline = df_actuals.iloc[-1:].append(df_subline.dropna().iloc[:1])

    for i, _ in enumerate(df_subline.columns):
        df_subline.iloc[0, i] = df_actuals.iloc[-1:, 0]

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_subline.index, df_subline.index[::-1]]),
            y=np.concatenate([df_subline["upper"], df_subline["lower"][::-1]]),
            marker_color="rgba(255,192,0,0.0)",
            fill="toself",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_subline.index,
            y=round(df_subline["actual"], 1),
            marker_color="rgba(255,192,0,0.9)",
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_subline.index,
            y=round(df_subline1["actual"], 1),
            marker_color="rgba(98,249,252,0.9)",
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ================================ GENERATE THE CONFIDENCE INTERVALLS ================================
    if conf_intervall:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_test.index, df_test.index[::-1]]),
                y=np.concatenate([df_test["upper"], df_test["lower"][::-1]]),
                name="Forecast CI*",
                fill="toself",
                hoverinfo="skip",
                marker_color="rgba(98,249,252,0.0)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [df_test_simulation.index, df_test_simulation.index[::-1]]
                ),
                y=np.concatenate(
                    [df_test_simulation["upper"], df_test_simulation["lower"][::-1]]
                ),
                name="Simulation CI*",
                fill="toself",
                hoverinfo="skip",
                marker_color="rgba(255,192,0,0.0)",
            )
        )

    # ================================ FORECAST, SIMULATION AND ACTUALS ================================
    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["actual"], 1),
            name="Actuals",
            marker_color="rgba(242,242,242,1)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test.index,
            y=round(df_test["forecast"], 1),
            name="Forecast",
            marker_color="rgba(98,249,252,0.9)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test_simulation.index,
            y=round(df_test_simulation["forecast"], 1),
            name="Simulation",
            marker_color="rgba(255,192,0,0.9)",
        )
    )

    # ================================ PLOT AESTHETICS ================================
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.0, x=1),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=50, b=50),
        plot_bgcolor="#151934",
        paper_bgcolor="#151934",
    )

    if display_fig == True:
        fig.show()
    else:
        return fig
