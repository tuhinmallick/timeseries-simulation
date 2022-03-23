import os,sys, logging, pathlib,pickle,traceback,time
src_location = pathlib.Path(__file__).absolute().parent
config_location = os.path.join(pathlib.Path(__file__).absolute().parent.parent, 'configs')
artifact_location = os.path.join(pathlib.Path(__file__).absolute().parent.parent, 'artifacts')
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))

# import streamlit_authenticator as stauth
# import extra_streamlit_components as stx
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx

# import streamlit_authenticator as stauth


import pandas as pd
from PIL import Image

# Importing the functionality package
from func.eda import exploratory_data_analysis
from func.tech_indicator import technical_indicator
from func.hist_forecast import historical_forecast
from func.simulation import get_simulation
from func.contact_form import contact
from func.sentiment_analyser import sentiment
from func.streamlit_visual import reedit_sentiments
from  lib.login.login_cred import login as _login
from  lib.login.authenticator import Hasher, Authenticate
# from lib.inputs.dataset import input_dataset
# from  lib.login.get_login_status import get_login_info as login

# from lib.inputs.dataset import ( input_columns,
#                                 input_dataset)
# from lib.misc.load import load_config, load_image
# from lib.dataprep.format import (
#     add_cap_and_floor_cols,
#     check_dataset_size,
#     filter_and_aggregate_df,
#     format_date_and_target,
#     format_datetime,
#     print_empty_cols,
#     print_removed_cols,
#     remove_empty_cols,
#     resample_df,
# )
st.set_page_config(page_title='Forecasty.Ai', layout='wide', page_icon="chart_with_upwards_trend",)
print('ping')
# Load config
# config, instructions, readme = load_config(
#     "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
# )
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
# import pdb;pdb.set_trace()
# if st.session_state['logged'] == False:
forecasty_image = Image.open(os.path.join(artifact_location, "logo", "logo_forecasty.PNG"))
st.sidebar.image(forecasty_image, output_format ='PNG', use_column_width  ='always')

# if st.experimental_get_query_params() == {}:
#         print('got') 
#         st.experimental_set_query_params(logged=False, cred =True,name='')
# if  st.experimental_get_query_params()['logged'][0] =='False':
#     names, usernames, passwords = _login.login_crediatils()
#     hashed_passwords = stauth.Hasher(passwords).generate()
#     authenticator = Authenticate(names,usernames,hashed_passwords,
#     'some_cookie_name','some_signature_key',cookie_expiry_days=30)
#     name, authentication_status, username = authenticator.login('Login','main')
# if st.experimental_get_query_params()['logged'][0] =='True' and st.experimental_get_query_params() != {}:
#             cookie_manager = stx.CookieManager(key='getout')
#             col1,col2 = st.sidebar.columns([1, 0.5])
#             col1.write('Welcome *%s*' % (st.experimental_get_query_params()['name'][0]))
#             if col2.button('Logout'):
#                 cookie_manager.delete('some_cookie_name')
#                 st.session_state['logout'] = True
#                 st.session_state['name'] = None
#                 st.session_state['username'] = None
#                 st.session_state['authentication_status'] = None
#                 st.experimental_set_query_params(logged=False, cred =True)
_login.set_session_state()
if st.session_state['authentication_status'] ==  False or st.session_state['authentication_status'] is None:
    names, usernames, passwords = _login.login_crediatils()
    hashed_passwords = Hasher(passwords).generate()
    authenticator = Authenticate(names,usernames,hashed_passwords,
        'some_cookie_name','some_signature_key',cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login','main')
    _login.set_name(name)
    _login.set_authentication_status(authentication_status)



@st.experimental_memo
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
    return df

def upload():
    dashboards = ('PGM_curated.csv','PGM_combined_tidy.csv')
    load_options = dict()
    load_options["toy_dataset"] = st.checkbox(
        "Load a uploaded dataset", True, help='Select this option if you want to work with uploaded Dataset'
    )
    if load_options["toy_dataset"]:
        dataset_name = st.selectbox(
            "Select a uploaded dataset",
            options=dashboards,
            help='Select the dataset you want to work with',
        )
        df = load_data(os.path.join(artifact_location,'source_data',dataset_name))
        st.write("{} has been uploaded".format(dataset_name))
    else :
        df =None
        try : 
            uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                df = load_data(uploaded_file)
                st.write("{} has been uploaded".format(uploaded_file.name))
        except Exception as err:
            st.write("{} is not the proper file format".format(uploaded_file.name))
    return df




def streamlit_menu():
    selected = option_menu(
            menu_title=None,  # required
            options=['Technical indicator(Beta)', 'Exploratory Data analysis', 'Base Forecast', 'Simulation','Sentiment Analyser(Beta)','Support'],  # required
            icons=["bar-chart-fill","table", "cash-coin", "calculator",'search','envelope'],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#151934"},
                "icon": {"color": "orange", "font-size": "15px"},
                "nav-link": {
                    "font-size": "12px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#264f27"},
            },
        )
    print(st.session_state['authentication_status'])
    head_col1.title(" PGM Price Forecasting")   
    head_col2.image(company_image, output_format ='PNG', use_column_width  ='auto')
    st.session_state.functionality_type = selected
    if st.session_state['authentication_status'] == True:
            col1,col2 = st.sidebar.columns([1, 0.5])
            col1.write('Welcome *%s*' % (st.session_state['name']))
            if col2.button('Logout', help= 'If the logout buttton does not work, please refresh the page'):
                cookie_manager = stx.CookieManager(key='logout')
                cookie_manager.delete('some_cookie_name')
                st.session_state['logout'] = True
                st.session_state['name'] = None
                st.session_state['username'] = None
                st.session_state['authentication_status'] = None
    return selected


if  st.session_state['authentication_status']:
    try : 
            data= None
            head_col1, head_col2 = st.columns([1,0.5])

            company_image = Image.open(os.path.join(artifact_location, "logo", "logo_basf.png"))
            # Forecasty logo
            dashboards = ('Technical indicator(Beta)', 'Exploratory Data analysis', 'Historical Forecast', 'Simulation')
            # option = st.empty()
            # option = st.selectbox('Please select the functionality : ',dashboards,key = 'functionality_typ',  on_change=_login.set_functionality_type, args = (option,))
            option = streamlit_menu()
            print(option)
            print(st.session_state.functionality_type)
            # functionality =st.form_submit_button('Apply', on_click=set_functionality_type, args = (option,))   
            if st.session_state.functionality_type != 'Historical Forecast' and st.session_state.functionality_type != []:
                with st.sidebar.expander("Dataset", expanded=True):
                    # data, load_options, config, datasets = input_dataset(config, readme, instructions)
                    # data, empty_cols = remove_empty_cols(data)
                    # print_empty_cols(empty_cols)
                        with st.spinner('Uploading the document'):
                            data = upload()
            if data is not None:
                if option == 'Technical indicator(Beta)':
                    technical_indicator(data)
                elif option == 'Exploratory Data analysis':
                    exploratory_data_analysis(data)
                elif option == 'Base Forecast':
                    historical_forecast()
                elif option == 'Simulation':
                    get_simulation(data)
                elif option == 'Support':
                    contact()
                elif option == 'Sentiment Analyser(Beta)':
                    with st.sidebar:
                        sentiment_type = option_menu("Sentiment Menu", ["Tweets", 'Reddit'], icons=['twitter', 'reddit'], menu_icon="search-heart", default_index=1,
                            styles={
                                    "container": {"padding": "0!important", "background-color": "#151934"},
                                    "icon": {"color": "orange", "font-size": "15px"},
                                    "nav-link": {
                                        "font-size": "12px",
                                        "text-align": "left",
                                        "margin": "0px",
                                        "--hover-color": "#eee",
                                    },
                                    "nav-link-selected": {"background-color": "#264f27"},
                                },)
                    if sentiment_type == 'Tweets':
                            sentiment()
                    elif sentiment_type == 'Reddit':
                            reedit_sentiments()
    except Exception as err:
                traceback.print_exc()
                print(err.args)
elif authentication_status == False:
        st.error('Username/password is incorrect')
elif authentication_status == None:
        st.warning('Please enter your username and password')
