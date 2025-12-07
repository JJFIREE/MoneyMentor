import streamlit as st
import auth_functions

# Log In form
st.title("Log In")
with st.form(key="login_form"):
    email = st.text_input("Email", placeholder="your.email@example.com")
    password = st.text_input("Password", type="password")
    submit_button = st.form_submit_button("Log In")
    st.page_link('user_options/forgot_password_pg.py', label='Reset Password')
    
    if submit_button:
        if email and password:
            with st.spinner('Logging in...'):
                # sign_in now returns boolean - handle it properly
                success = auth_functions.sign_in(email, password)
                
                if success:
                    st.success('Logged in successfully!')
                    st.rerun()  # Refresh to load the main app
                else:
                    # Error message is already set in session state by sign_in function
                    if 'auth_warning' in st.session_state:
                        st.error(st.session_state.auth_warning)
        else:
            st.error("Please provide both email and password.")

# Show any auth messages that might be set
if 'auth_success' in st.session_state and st.session_state.auth_success:
    st.success(st.session_state.auth_success)
if 'auth_warning' in st.session_state and st.session_state.auth_warning:
    st.error(st.session_state.auth_warning)