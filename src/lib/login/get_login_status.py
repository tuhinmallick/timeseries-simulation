import streamlit as st
import os,sys, logging, pathlib,pickle,traceback
src_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
import extra_streamlit_components as stx
from  login.authenticator import Hasher, Authenticate
from  login.login_cred import login as _login
def get_login_info():
    if st.experimental_get_query_params() == {}:
        print('got') 
        st.experimental_set_query_params(logged=False, cred =True,name='')
    if  st.experimental_get_query_params()['logged'][0] =='False':
        names, usernames, passwords = _login.login_crediatils()
        hashed_passwords = Hasher(passwords).generate()
        authenticator = Authenticate(names,usernames,hashed_passwords,
        'some_cookie_name','some_signature_key',cookie_expiry_days=30)
        name, authentication_status, username = authenticator.login('Login','main')
    if st.experimental_get_query_params()['logged'][0] =='True' and st.experimental_get_query_params() != {}:
        cookie_manager = stx.CookieManager(key='getout')
        col1,col2 = st.sidebar.columns([1, 0.5])
        col1.write(f"Welcome *{st.experimental_get_query_params()['name'][0]}*")
        if col2.button('Logout'):
            cookie_manager.delete('some_cookie_name')
            st.session_state['logout'] = True
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.session_state['authentication_status'] = None
            st.experimental_set_query_params(logged=False, cred =True)

    if  st.experimental_get_query_params()['logged'][0] =='True':
        st.title('Some content')
    elif st.experimental_get_query_params()['cred'][0] =='False' :
        st.error('Username/password is incorrect. Please check your credentials or contact support.')
    return name, authentication_status, username



