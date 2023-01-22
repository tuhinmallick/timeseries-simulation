import streamlit as st
import os, sys, logging, pathlib, pickle, traceback

src_location = pathlib.Path(__file__).absolute().parent.parent
artifact_location = os.path.join(
    pathlib.Path(__file__).absolute().parent.parent.parent, "artifacts"
)
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from lib.login.login_cred import login
import utils.eda_base as eda_base


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   Historical Forecast
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


def historical_forecast():
    item_dict = {}
    commodity = st.sidebar.radio(
        ("Select your commodity among PGM"), ("Platinum", "Palladium", "Rhodium")
    )
    forecast_list = ["forecast_chart", "forecast_drivers", "forecast_backtesting"]
    for items in forecast_list:
        pickle_df = f"{commodity}_{items}.pickle"
        with open(
            os.path.join(artifact_location, "forecast", items, pickle_df), "rb"
        ) as _file:
            item = pickle.load(_file)
            item_dict[items] = item

    forecast, feature_importance, backtesting = st.sidebar.columns([0.8, 0.7, 1])

    if forecast.button("Forecast", on_click=login.set_Forecast):
        st.write(f"### The plot for Forecast of {commodity} ")
        st.plotly_chart(
            item_dict.get("forecast_chart"),
            use_container_width=False,
            sharing="streamlit",
        )
        st.session_state.Forecast = False

    if feature_importance.button("Drivers", on_click=login.set_Drivers):
        st.write(f"### The plot for Drivers of {commodity} ")
        st.plotly_chart(
            item_dict.get("forecast_drivers"),
            use_container_width=False,
            sharing="streamlit",
        )
        st.session_state.Drivers = False

    if backtesting.button("Backtesting", on_click=login.set_Backtesting):
        st.write(f"### The plot for Backtesting of {commodity} ")
        st.plotly_chart(
            item_dict.get("forecast_backtesting"),
            use_container_width=False,
            sharing="streamlit",
        )
        st.session_state.Backtesting = False
