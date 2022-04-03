import os, sys, logging, pathlib, pickle, json
import streamlit as st
import pandas as pd

src_location = pathlib.Path(__file__).absolute().parent.parent
ts_location = pathlib.Path(__file__).absolute().parent.parent.parent
artifact_location = os.path.join(ts_location, "artifacts")
animation_path = os.path.join(ts_location, "artifacts", "animation")
utils_path = os.path.join(pathlib.Path(__file__).absolute().parent.parent, "lib")
# if os.path.realpath(utils_path) not in sys.path:
#     sys.path.append(os.path.realpath(utils_path))
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from streamlit_lottie import st_lottie_spinner
from lib.login.login_cred import login
import run_simulate as simulate


@st.experimental_memo
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_name = "simulation_animation.json"
lottie_json = load_lottiefile(os.path.join(animation_path, lottie_name))
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   Simulation
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def plot_simulation(
    simulation_fig,
    forecast,
    original_forecast,
    horizon,
    commodity,
    data=None,
    final_simulation_correlation_df=None,
):
    diff = forecast.reset_index()[["forecast", "lower", "upper"]].sub(
        original_forecast.reset_index()[["forecast", "lower", "upper"]]
    )
    with st.expander("Simulated Forecast Plot"):
        st.write(
            f"#### The plot of the simulated forecast of {commodity} for {horizon} months ahead forecast"
        )
        st.plotly_chart(simulation_fig, use_container_width=False, sharing="streamlit")
    with st.expander("Predicted Forecast chart"):
        st.write(
            "####  Original monthly average forecast prices for {} months".format(
                horizon
            )
        )
        original_forecast = original_forecast.rename(
            columns={
                "forecast": "Predicted price(USD/Toz)",
                "lower": "Lower probable price(USD/Toz)",
                "upper": "Upper probable price(USD/Toz)",
            }
        )
        st.dataframe(original_forecast)
    with st.expander("Simulated Forecast chart"):
        st.write(
            "#### Simulated monthly average forecast prices for {} months".format(
                horizon
            )
        )
        diff = diff.rename(
            columns={
                "forecast": "Difference in  price(USD/Toz)",
                "lower": "Difference in Lower probable price(USD/Toz)",
                "upper": "Difference in Upper probable price(USD/Toz)",
            }
        )
        diff.index = forecast.index
        forecast = forecast.rename(
            columns={
                "forecast": "Simulated price(USD/Toz)",
                "lower": "Lower probable price(USD/Toz)",
                "upper": "Upper probable price(USD/Toz)",
            }
        )
        simulated_forecast = pd.concat([forecast, diff], axis=1, ignore_index=False)
        st.dataframe(simulated_forecast)
    with st.expander("Feature Importance chart"):
        pickle_df = f"{commodity}_features_fig.pickle"
        # st.write("**Please note** :  The features listed in this chart shows it influences the prices of the commodity that is being predicted. The new comodity that ")
        with open(
            os.path.join(artifact_location, "forecast", "forecast_drivers", pickle_df),
            "rb",
        ) as meta_features:
            drivers_chart = pickle.load(meta_features)
        st.write(f"### The plot for feature importance of {commodity} ")
        st.plotly_chart(drivers_chart, use_container_width=False, sharing="streamlit")
        st.write(
            "*ma = Moving Average, ema = Exp. Moving Average, momentum = Price momentum while suffix like 3 means the number of months"
        )
    with st.expander("Correlation Fraction chart"):
        st.dataframe(final_simulation_correlation_df)
    return forecast


def get_simulation(data):
    simulation_dict = {}
    simulation_fig = {}
    commodity = st.sidebar.radio(
        ("Select your commodity among PGM"), ("Platinum", "Palladium", "Rhodium")
    )
    simulation_options = st.sidebar.form("simulation-options")
    with simulation_options:
        # commodity, num_sim_feat, perc_change = st.empty(), st.empty(), st.empty()
        all_features = data.columns
        num_sim_feat = st.sidebar.number_input(
            label="Please select the number of features to simulate",
            min_value=1,
            max_value=len(all_features),
            key="num_sim_feat",
        )
        horizon = st.sidebar.number_input(
            label="Please select the number of lookahead months to simulate",
            min_value=1,
            max_value=3,
            key="horizon",
        )
        for n in range(1, num_sim_feat + 1):
            sim_target = st.empty()
            sim_target = st.sidebar.selectbox(
                "Please select the target feature to simulate : ",
                all_features,
                key="sim_feat{}".format(n),
            )
            perc_change = st.sidebar.slider(
                "Percent Change in value",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="sim_prec_feat{}".format(n),
            )
            # corr_btn = st.sidebar.checkbox('Check correlation matrix', on_change= set_corr_target,args=(sim_target,), key =n)
            # simulation_dict =  dict([(sim_target,perc_change)])
            col1, col2 = st.sidebar.columns([1, 1])
            col1.write(f"Current value: {round(data[sim_target][-1])}")
            col2.write(
                f"New value: {round(data[sim_target][-1]+data[sim_target][-1]*(perc_change/100))}"
            )
            simulation_dict.update({sim_target: perc_change})
        simulated = st.form_submit_button(
            "Simulate 🚀", on_click=login.set_simulation_dict, args=(simulation_dict,)
        )
    if simulated:
        with st_lottie_spinner(lottie_json, quality="high"):
            # with st.spinner(text="Simulating the future"):
            (
                simulation_fig,
                forecast,
                original_forecast,
                final_simulation_correlation_df,
            ) = simulate.main(
                simulation_dict=st.session_state.simulation_dict,
                target=commodity,
                sim_df=data,
                horizon=horizon,
            )
        plotiing = plot_simulation(
            simulation_fig,
            forecast,
            original_forecast,
            horizon,
            commodity,
            data,
            final_simulation_correlation_df,
        )
