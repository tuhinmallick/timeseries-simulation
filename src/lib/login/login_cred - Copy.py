import streamlit as st

class login ():
        def login_crediatils():
                names = ['Cathal Prendergast', 'Paul Moschella', 'Bret Mantone','Kaan Kaymak', 'Matthias Dohrn', 'Kate Silvestri',
                        'Scott M Mizrahi', 'Amanda Colyer', 'William Kaplowitz', 'James Gove', 'Vas Vergopoulos', 'Stephen Pender','Will Thomas','Toby Green','Matthew E Gidicsin','Pascal Ochs','Rahul Kalippurayil Moozhipurath','Erika Fonseca','Tuhin Mallick', 'Ralph Debusmann','John Metcalf']
                usernames =  ['Cathal Prendergast', 'Paul Moschella', 'Bret Mantone','Kaan Kaymak', 'Matthias Dohrn', 'Kate Silvestri',
                        'Scott M Mizrahi',  'Amanda Colyer', 'William Kaplowitz', 'James Gove', 'Vas Vergopoulos', 'Stephen Pender','Will Thomas','Toby Green','Matthew E Gidicsin','Pascal Ochs','Rahul Kalippurayil Moozhipurath','Erika Fonseca','Tuhin Mallick', 'Ralph Debusmann', 'John Metcalf']
                passwords = ['YiSA2XNLjlVgwuX', 
                        'pyK7If0ICcb4rTD', 
                        'EQ0r39iRNuCUNAC',
                        'V4dVwhS5DiJpXIe', 
                        'T4zU73l5gQ2doF0',
                        'E2EAK6aJWkgLnJM', 
                        'QrxqKzIfKPaj1fa', 
                        'kyqo4OoWi4oYLUR', 
                        'gmBIVYYHal2rNqh', 
                        'Rvl5naCEXbgyy2b', 
                        'ESZztihHsGI02Bu', 
                        'rdQnHRh3Dy0czdU',
                        'QHcUXlnD6wPYxjQ',
                        'GhOl6tN5RtNOf4Y',
                        'Z1W3HWzsygd9vOw',
                        '2WLVoJ3ylx6NHZ4',
                        '6VcZwzTEERlFNko',
                        'f94pTEW6XO1Yljv',
                        'qwcS732MfbD6YCc',
                        'IS3bppMAilySrZd',
                        'RDBig7nz8ZFzww5']
                return names, usernames, passwords

        def set_session_state():
                # default values
                if 'logged' not in st.session_state:
                        st.session_state.logged = False
                if 'functionality_type' not in st.session_state:
                        st.session_state.functionality_type = []
                if 'num_sim_feat' not in st.session_state:
                        st.session_state.num_sim_feat = 0
                if 'horizon' not in st.session_state:
                        st.session_state.horizon = 0
                if 'Forecast' not in st.session_state:
                        st.session_state.Forecast = False
                if 'Drivers' not in st.session_state:
                        st.session_state.Drivers = False
                if 'Backtesting' not in st.session_state:
                        st.session_state.Backtesting = False
                if 'simulation_dict' not in st.session_state:
                        st.session_state.simulation_dict = {}
                if 'corr_target' not in st.session_state:
                        st.session_state.corr_target = ''
                if 'authentication_status' not in st.session_state:
                        st.session_state.authentication_status = False
                if 'name' not in st.session_state:
                        st.session_state.name = ''

        def set_corr_target(self):
                st.session_state.corr_target = self
        def set_number_of_sim_feat(self):
                st.session_state.num_sim_feat = self
        def set_horizon(self):
                st.session_state.horizon = self
        def set_functionality_type(self):
                st.session_state.functionality_type = self
        def set_simulation_dict(self):
                st.session_state.simulation_dict = self
        def set_Forecast():
                st.session_state.Forecast = True
        def set_Drivers():
                st.session_state.Drivers = True
        def set_Backtesting():
                st.session_state.Backtesting = True
        def set_authentication_status(self):
                st.session_state.authentication_status = self
        def set_name(self):
                st.session_state.name = self
        def set_login_status():
                st.session_state.logged = True
                