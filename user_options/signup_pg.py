import streamlit as st
from auth_functions import *

# Create Account form
st.title("Create Account")
st.write("Create your MoneyMentor account to get started")

with st.form(key="create_account_form"):
    email = st.text_input("Email", placeholder="your.email@example.com")
    password = st.text_input("Password", type="password", 
                           help="Password should be at least 6 characters long")
    submit_button = st.form_submit_button("Sign Up")

if submit_button:
    if email and password:
        with st.spinner('Creating account...'):
            success = create_account(email, password)
            
            if success:                
                # Display the success message
                st.success('Account Created Successfully! 🎉')
                st.success('Check your inbox to verify your email before signing in')
                st.page_link('user_options/login_pg.py', label='Go to Login')
                
                # Clear any existing warnings
                if 'auth_warning' in st.session_state:
                    st.session_state.auth_warning = None
            else:
                # Show the warning from session state
                if 'auth_warning' in st.session_state and st.session_state.auth_warning:
                    st.error(st.session_state.auth_warning)
                else:
                    st.error("Account creation failed. Please try again.")
    else:
        st.error("Please provide both email and password.")

# Display any auth messages that might be set (in case they're not shown above)
if 'auth_success' in st.session_state and st.session_state.auth_success:
    st.success(st.session_state.auth_success)
    
if 'auth_warning' in st.session_state and st.session_state.auth_warning:
    st.error(st.session_state.auth_warning)