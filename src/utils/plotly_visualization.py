import pandas as pd
import numpy as np 

import plotly.graph_objects as go

def plotly_plot_forecast(df_actuals, df_forecast, horizon=6, display_fig=True, figsize=(1200, 600), **kwargs):
    """ Function to generate the forecast plot with actuals and confidence intervall.

    Args:
        df_actuals (pandas.DataFrame): Actual values of the target series with date on index. Need column name "actual".
        df_forecast (pandas.DataFrame): Forecasted values and confidence intervalls of the target series with date on index. Need column name "forecast", "lower", "upper".
        horizon (int, optional): Horizon value to fix adjust the combined dataframe and to make line consistent. Defaults to 6.
        display_fig (bool, optional): Select if plot is displayed or figure object is returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the plotly figure object if display_fig is True.
    """
    
    df_test = pd.concat([df_actuals, df_forecast], axis=1)

    # Fix the latest value so the lines are connected.
    for i in range(1, len(df_test.columns)):
        df_test.iloc[-horizon-1, i] = df_test.iloc[-horizon-1, 0]
        
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df_test.index, y=df_test["actual"], name="Actuals")
        )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_test.index, df_test.index[::-1]]),
            y=np.concatenate([df_test["upper"], df_test["lower"][::-1]]),
            name="Confidence Interval",
            fill='toself',
            hoverinfo="skip")
        )

    fig.add_trace(
        go.Scatter(x=df_test.index, y=df_test["forecast"], name="Forecast")
        )

    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x",
        legend=dict(
            yanchor="top",
            xanchor="right"
            ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=30, b=50),
        plot_bgcolor='#151934',
        paper_bgcolor ='#151934'
        # gridcolor=
        )
    
    if display_fig == True:
        fig.show()
    else: 
        return fig
    
def plotly_plot_backtest(df_backtest, display_fig=True, figsize=(1200, 600), **kwargs):
    """ Function to generate the backtest plot with actuals, forecast and confidence intervall.

    Args:
        df_backtest (pandas.DataFrame): Backtesting results with date on the index. 
        display_fig (bool, optional): Select if plot is displayed or figure object is returned. Defaults to True.
        figisze (tuple, optional): Select the width and height of the figure.

    Returns:
        plotly figure object: Returns the plotly figure object if display_fig is True.
    """
    fig = go.Figure()

    # NOTE: Until fixed add confidence first so that the other traces lay on top and not below.
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_backtest.index, df_backtest.index[::-1]]),
            y=np.concatenate([df_backtest["upper"], df_backtest["lower"][::-1]]),
            name="Confidence Interval",
            fill='toself',
            hoverinfo="skip")
        )
    
    fig.add_trace(
        go.Scatter(x=df_backtest.index, y=df_backtest["actual"], name="Actuals")
        )

    fig.add_trace(
        go.Scatter(x=df_backtest.index, y=df_backtest["forecast"], name="Forecast")
        )

    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x",
        legend=dict(
            yanchor="top",
            xanchor="right"
            ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=30, b=50),
        plot_bgcolor='#151934',
        paper_bgcolor ='#151934'
        )
    
    if display_fig == True:
        fig.show()
    else: 
        return fig
    
    
def plotly_feature_importance(df_features, number_features=10, display_fig=True, figsize=(1200, 600), **kwargs):
    """ Function to plot the feature importance of the model.

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

    fig.add_trace(
        go.Bar(y=df_features.loc[:number_features, col_names[0]], x=df_features.loc[:number_features, col_names[1]], orientation="h", marker_color="rgba(98,249,252,0.9)")
        )

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
        plot_bgcolor='#151934',
        paper_bgcolor ='#151934'
        )

    if display_fig == True:
        fig.show()
    else: 
        return fig
    
def plotly_plot_simulation(df_actuals, df_forecast, df_simulation, horizon=3, conf_intervall=True, display_fig=True, figsize=(1200, 600), **kwargs):
    """ Function to generate the forecast plot with actuals and confidence intervall.

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

    # Fix the latest value so the lines are connected.
    for i in range(1, len(df_test.columns)):
        df_test.iloc[-horizon-1, i] = df_test.iloc[-horizon-1, 0]
        
    # Fix the latest value so the lines are connected.
    for i in range(1, len(df_test_simulation.columns)):
        df_test_simulation.iloc[-horizon-1, i] = df_test_simulation.iloc[-horizon-1, 0]
        
    fig = go.Figure()
    
    
    if conf_intervall:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_test.index, df_test.index[::-1]]),
                y=np.concatenate([df_test["upper"], df_test["lower"][::-1]]),
                name="Confidence Interval Forecast",
                fill='toself',
                hoverinfo="skip")
            )
        
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_simulation.index, df_simulation.index[::-1]]),
                y=np.concatenate([df_simulation["upper"], df_simulation["lower"][::-1]]),
                name="Confidence Interval Simulation ",
                fill='toself',
                hoverinfo="skip")
            )

    fig.add_trace(
        go.Scatter(x=df_test.index, y=df_test["actual"], name="Actuals")
        )
    
    fig.add_trace(
        go.Scatter(x=df_test.index, y=df_test["forecast"], name="Forecast")
        )

    fig.add_trace(
        go.Scatter(x=df_test_simulation.index, y=df_test_simulation["forecast"], name="Simulation")
        )

    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        xaxis_title="",
        yaxis_title=f"Price, USD/toz",
        hovermode="x",
        legend=dict(
            yanchor="top",
            xanchor="right"
            ),
        template="plotly_dark",
        margin=dict(l=80, r=30, t=30, b=50),
        plot_bgcolor='#151934',
        paper_bgcolor ='#151934'
        # gridcolor=
        )
    
    if display_fig == True:
        fig.show()
    else: 
        return fig