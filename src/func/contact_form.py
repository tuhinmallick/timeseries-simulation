import os, sys, logging, pathlib, pickle, traceback

func_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(func_location) not in sys.path:
    sys.path.append(os.path.realpath(func_location))
import streamlit as st  # pip install streamlit


def contact():
    st.header(":mailbox: Get in touch with us")

    contact_form = """
<form action="https://formsubmit.co/tuhin.mllk@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""

    st.markdown(contact_form, unsafe_allow_html=True)
    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css(os.path.join(func_location, "style", "style.css"))
    
    
