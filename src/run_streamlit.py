import requests
import os, sys, logging, pathlib, pickle, traceback
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import streamlit_authenticator as stauth

import pandas as pd
from PIL import Image

from lib.login.login_cred import login as _login

src_location = pathlib.Path(__file__).absolute().parent
config_location = os.path.join(
    pathlib.Path(__file__).absolute().parent.parent, "configs"
)
artifact_location = os.path.join(
    pathlib.Path(__file__).absolute().parent.parent, "artifacts"
)
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
import utils.eda_base as eda_base
import run_simulate as simulate


def __init__(self):
    self.bytes_data = None


@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
    return df


def upload():
    df = None
    try:
        uploaded_files = st.file_uploader(
            "Choose a CSV file", accept_multiple_files=True
        )
        for uploaded_file in uploaded_files:
            df = load_data(uploaded_file)
            st.write("{} has been uploaded".format(uploaded_file.name))
    except Exception as err:
        st.write("{} is not the proper file format".format(uploaded_file.name))
    return df


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   Technical Indicator
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def technical_indicator(data):
    st.write("# Coming soon")


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   EDA
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def exploratory_data_analysis(data):
    all_features = data.columns
    commodity = st.sidebar.selectbox(
        "Please select the target feature : ", all_features, key="EDA_target_feature"
    )
    target_name = commodity
    # target_name = f"{commodity}_spot_price"
    eda = None
    option = st.sidebar.selectbox(
        "Please select the type of plot : ",
        ["Time Series", "Cross-Correlation", "Box Plot", "Granger Casaulty check"],
        key="EDA_type",
    )
    st.sidebar.write("You have selected:", option)
    if data is not None:
        st.write("##### You have selected the commodity: ", commodity)
        eda = eda_base.exploratory_data_analysis(target_name=target_name, df=data)
        if option == "Time Series":
            fig = eda.single_timeseries_plot(
                target_name, True, True, streamlit=True, transparent=False
            )
            st.pyplot(fig, transparent=False)
        elif option == "Cross-Correlation":
            features = data.drop(target_name, axis=1).columns
            feat = st.sidebar.selectbox(
                "Please select the feature you want to correlate : ",
                features,
                key="Cross_Correlation",
            )
            fig = eda.single_correlate_plot(
                y_variable=target_name,
                x_variable=feat,
                figsize=(20, 14),
                file_name_addition=4,
                plot_transparent_backgorund=False,
                streamlit=True,
                fontsize_title=40,
                fontsize_label=40,
            )
            st.pyplot(fig, transparent=False)
        elif option == "Box Plot":
            fig = eda.seasonal_boxplot_ym(
                y_variable=target_name,
                streamlit=True,
                plot_transparent_backgorund=False,
            )
            st.pyplot(fig, transparent=False)
        elif option == "Granger Casaulty check":
            features = data.drop(target_name, axis=1).columns
            feat = st.sidebar.selectbox(
                "Please select the feature you want to check: ", features, key="Granger"
            )
            fig = eda.single_granger_plot(
                y_variable=target_name,
                x_variable=feat,
                max_lags=30,
                streamlit=True,
                save_path=None,
                figsize=(15, 6),
                dpi=180,
                plot_transparent_backgorund=False,
                fontsize_title=20,
                fontsize_label=20,
                fontsize_xyticks=20,
            )
            st.pyplot(fig, transparent=False)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   Historical Forecast
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def plot_historical_forecast(chart, commodity):
    st.write(
        f"### The plot for {st.session_state.historical_func_type} of {commodity} "
    )
    st.plotly_chart(chart, use_container_width=False, sharing="streamlit")


def historical_forecast():
    commodity = st.sidebar.radio(
        ("Select your commodity among PGM"), ("Platinum", "Palladium", "Rhodium")
    )
    forecast, feature_importance, backtesting = st.sidebar.columns([0.8, 0.7, 1])
    with forecast:
        st.button("Forecast", on_click=set_Forecast)
        pickle_df = f"{commodity}_forecast_fig.pickle"
        with open(
            os.path.join(artifact_location, "forecast", "forecast_chart", pickle_df),
            "rb",
        ) as _file:
            forecast_chart = pickle.load(_file)
    with feature_importance:
        st.button("Drivers", on_click=set_Drivers)
        pickle_df = f"{commodity}_features_fig.pickle"
        with open(
            os.path.join(artifact_location, "forecast", "forecast_drivers", pickle_df),
            "rb",
        ) as _file:
            drivers_chart = pickle.load(_file)
    with backtesting:
        st.button("Backtesting", on_click=set_Backtesting)
        pickle_df = f"{commodity}_backtest_fig.pickle"
        with open(
            os.path.join(
                artifact_location, "forecast", "forecast_backtesting", pickle_df
            ),
            "rb",
        ) as _file:
            backtesting_chart = pickle.load(_file)
    if st.session_state.Forecast == True:
        st.write("### The plot for Forecast of {} ".format(commodity))
        st.plotly_chart(forecast_chart, use_container_width=False, sharing="streamlit")
        st.session_state.Forecast = False
    if st.session_state.Drivers == True:
        st.write("### The plot for Drivers of {} ".format(commodity))
        st.plotly_chart(drivers_chart, use_container_width=False, sharing="streamlit")
        st.session_state.Drivers = False
    if st.session_state.Backtesting == True:
        st.write("### The plot for Backtesting of {} ".format(commodity))
        st.plotly_chart(
            backtesting_chart, use_container_width=False, sharing="streamlit"
        )
        st.session_state.Backtesting = False


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
        st.write("####  Original  forecast prices for {} months".format(horizon))
        original_forecast = original_forecast.rename(
            columns={
                "forecast": "Predicted price($)",
                "lower": "Lower probable price($)",
                "upper": "Upper probable price($)",
            }
        )
        st.dataframe(original_forecast)
    with st.expander("Simulated Forecast chart"):
        st.write("#### Simulated forecast prices for {} months".format(horizon))
        diff = diff.rename(
            columns={
                "forecast": "Difference in  price($)",
                "lower": "Difference in Lower probable price($)",
                "upper": "Difference in Upper probable price($)",
            }
        )
        diff.index = forecast.index
        forecast = forecast.rename(
            columns={
                "forecast": "Simulated price($)",
                "lower": "Lower probable price($)",
                "upper": "Upper probable price($)",
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
    with st.expander("Correlation Fraction chart"):
        st.dataframe(final_simulation_correlation_df)
    return forecast


def simulation(data):
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
                "Percent Change in price",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="sim_prec_feat{}".format(n),
            )
            # corr_btn = st.sidebar.checkbox('Check correlation matrix', on_change= set_corr_target,args=(sim_target,), key =n)
            # simulation_dict =  dict([(sim_target,perc_change)])
            simulation_dict.update({sim_target: perc_change})
        simulated = st.form_submit_button(
            "Simulate", on_click=set_simulation_dict, args=(simulation_dict,)
        )
    lottie_url = "https://assets4.lottiefiles.com/packages/lf20_dews3j6m.json"
    lottie_json = load_lottieurl(lottie_url)
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


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   setting session statess
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


def set_session_state():
    # default values
    if "functionality_type" not in st.session_state:
        st.session_state.functionality_type = []
    if "num_sim_feat" not in st.session_state:
        st.session_state.num_sim_feat = 0
    if "horizon" not in st.session_state:
        st.session_state.horizon = 0
    if "Forecast" not in st.session_state:
        st.session_state.Forecast = False
    if "Drivers" not in st.session_state:
        st.session_state.Drivers = False
    if "Backtesting" not in st.session_state:
        st.session_state.Backtesting = False
    if "simulation_dict" not in st.session_state:
        st.session_state.simulation_dict = {}
    if "corr_target" not in st.session_state:
        st.session_state.corr_target = ""
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False
    if "name" not in st.session_state:
        st.session_state.name = ""


def set_corr_target(option: str):
    st.session_state.corr_target = option


def set_number_of_sim_feat(option: int):
    st.session_state.num_sim_feat = option


def set_horizon(option: int):
    st.session_state.horizon = option


def set_functionality_type(option: str):
    st.session_state.functionality_type = option


def set_simulation_dict(option: str):
    st.session_state.simulation_dict = option


def set_Forecast():
    st.session_state.Forecast = True


def set_Drivers():
    st.session_state.Drivers = True


def set_Backtesting():
    st.session_state.Backtesting = True


def set_authentication_status(option: bool):
    st.session_state.authentication_status = option


def set_name(option: str):
    st.session_state.name = option


def main():
    # def load_config(config_name):
    #     config =None
    #     config_path = os.path.join(config_location, config_name)
    #     with open(config_path) as file:
    #         try:
    #             config = yaml.safe_load(file)
    #         except yaml.YAMLError as exc:
    #             print(exc)
    #     return config
    # config = load_config("config.yaml")
    st.set_page_config(page_title="Forecasty.Ai", layout="wide")
    st.markdown(
        """ <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """,
        unsafe_allow_html=True,
    )
    padding = 0
    st.markdown(
        f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """,
        unsafe_allow_html=True,
    )
    names = [
        "Cathal Prendergast",
        "Paul Moschella",
        "Bret Mantone",
        "Kaan Kaymak",
        "Matthias Dohrn",
        "Kate Silvestri",
        "Scott M Mizrahi",
        "Amanda Colyer",
        "William Kaplowitz",
        "James Gove",
        "Vas Vergopoulos",
        "Stephen Pender",
        "Will Thomas",
        "Toby Green",
        "Matthew E Gidicsin",
        "Pascal Ochs",
        "Rahul Kalippurayil Moozhipurath",
        "Erika Fonseca",
        "Tuhin Mallick",
        "Ralph Debusmann",
        "John Metcalf",
    ]
    usernames = [
        "Cathal Prendergast",
        "Paul Moschella",
        "Bret Mantone",
        "Kaan Kaymak",
        "Matthias Dohrn",
        "Kate Silvestri",
        "Scott M Mizrahi",
        "Amanda Colyer",
        "William Kaplowitz",
        "James Gove",
        "Vas Vergopoulos",
        "Stephen Pender",
        "Will Thomas",
        "Toby Green",
        "Matthew E Gidicsin",
        "Pascal Ochs",
        "Rahul Kalippurayil Moozhipurath",
        "Erika Fonseca",
        "Tuhin Mallick",
        "Ralph Debusmann",
        "John Metcalf",
    ]
    passwords = [
        "YiSA2XNLjlVgwuX",
        "pyK7If0ICcb4rTD",
        "EQ0r39iRNuCUNAC",
        "V4dVwhS5DiJpXIe",
        "T4zU73l5gQ2doF0",
        "E2EAK6aJWkgLnJM",
        "QrxqKzIfKPaj1fa",
        "kyqo4OoWi4oYLUR",
        "gmBIVYYHal2rNqh",
        "Rvl5naCEXbgyy2b",
        "ESZztihHsGI02Bu",
        "rdQnHRh3Dy0czdU",
        "QHcUXlnD6wPYxjQ",
        "GhOl6tN5RtNOf4Y",
        "Z1W3HWzsygd9vOw",
        "2WLVoJ3ylx6NHZ4",
        "6VcZwzTEERlFNko",
        "f94pTEW6XO1Yljv",
        "tuhin",
        "IS3bppMAilySrZd",
        "RDBig7nz8ZFzww5",
    ]
    set_session_state()
    if (
        st.session_state["authentication_status"] == False
        or st.session_state["authentication_status"] is None
    ):
        names, usernames, passwords = _login.login_crediatils()
        hashed_passwords = stauth.Hasher(passwords).generate()
        credentials = {
            "usernames": {
                username: {"name": name, "password": password}
                for username, name, password in zip(usernames, names, hashed_passwords)
            }
        }
        authenticator = stauth.Authenticate(
            credentials,
            "some_cookie_name",
            "some_signature_key",
            30,
        )
        name, authentication_status, user_name = authenticator.login(
            "Login", location="main"
        )
        set_name(name)
        set_authentication_status(authentication_status)
    if st.session_state["authentication_status"]:
        try:
            data = None
            data_uploaded = False

            head_col1, head_col2 = st.columns([1, 0.5])
            image = Image.open(
                os.path.join(artifact_location, "logo", "logo_forecasty.PNG")
            )
            company_image = Image.open(
                os.path.join(artifact_location, "logo", "logo_basf.png")
            )
            head_col1.title(" PGM Price Forecasting")
            head_col2.image(company_image, output_format="PNG", use_column_width="auto")
            # Forecasty logo
            st.sidebar.image(image, output_format="PNG", use_column_width="always")
            # st.sidebar.write('##### Machine learns, company earns')
            st.sidebar.write("Welcome *%s*" % (st.session_state["name"]))
            uploading_options = st.sidebar.form("uploading-options")
            option = st.empty()
            option = st.selectbox(
                "Please select the functionality : ",
                [
                    "Technical indicator",
                    "Exploratory Data analysis",
                    "Base Forecast",
                    "Simulate the future",
                ],
                key="functionality_typ",
                on_change=set_functionality_type,
                args=(option,),
            )
            # functionality =st.form_submit_button('Apply', on_click=set_functionality_type, args = (option,))
            if (
                st.session_state.functionality_type != "Base Forecast"
                and st.session_state.functionality_type != []
            ):
                with uploading_options:
                    with st.spinner("Uploading the document"):
                        data = upload()
                        data_uploaded = True
                    st.form_submit_button("Upload")
            if data is not None:
                if option == "Technical indicator":
                    technical_indicator(data)
                elif option == "Exploratory Data analysis":
                    exploratory_data_analysis(data)
                elif option == "Base Forecast":
                    historical_forecast()
                elif option == "Simulate the future" and data is not None:
                    simulation(data)
        except Exception as err:
            traceback.print_exc()
            print(err.args)

    elif st.session_state["authentication_status"] == False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] == None:
        st.warning("Please enter your username and password")


if __name__ == "__main__":
    main()
