# pages/forgot_password.py
import streamlit as st
import auth_functions as auth_functions

st.title("Password Recovery")
st.write("Enter your email address and we'll send you a link to reset your password.")

with st.form(key="forgot_password_form"):
    email = st.text_input("Email", placeholder="your.email@example.com")
    submit_button = st.form_submit_button("Send Password Reset Link")
    
    if submit_button:
        if email:
            with st.spinner('Sending password reset link...'):
                try:
                    auth_functions.reset_password(email)
                    # The success message will be shown by the reset_password function
                    # via st.session_state.auth_success
                except Exception as e:
                    st.error(f"Failed to send reset email: {str(e)}")
        else:
            st.error("Please provide your email for password recovery.")

# Show any auth messages
if 'auth_success' in st.session_state:
    st.success(st.session_state.auth_success)
    # Clear the message after showing it
    st.session_state.auth_success = None

if 'auth_warning' in st.session_state:
    st.error(st.session_state.auth_warning)
    # Clear the message after showing it  
    st.session_state.auth_warning = None
