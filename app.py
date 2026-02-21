import streamlit as st
from auth_functions import *

# Initialize Firebase
initialize_firebase_once()

# Get the db reference from auth_functions
from auth_functions import db


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rate_us_button():
    @st.dialog("How do you rate our app?")
    def rate():
        sentiment_mapping = ["one", "two", "three", "four", "five"]
        selected = st.feedback("stars")
        if selected is not None:
            st.session_state.rate = {"item": sentiment_mapping[selected]}
            st.rerun()
    rate()
    return





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN APP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.set_page_config(
    page_title="Financial Agent", 
    page_icon="💰", 
    layout="wide", 
    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed')
)

st.logo(
    image = "assets/logo_finfriend.png",
    size = "large",
    icon_image = "assets/logo_finfriend.png",
)

st.markdown("""
    <style>
        /* Button */
        .stButton>button {
            background-color: rgba(131, 158, 101, 0.8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            width: 100%;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #839E65;
            transform: translateY(-2px);
        }
    </style>""", unsafe_allow_html=True)



if 'user_info' in st.session_state:
    user_info = st.session_state.user_info
    user_id = user_info.get('localId')  # Use .get() for safer access
    st.session_state.user_id = user_id
    
    # Debug print
    print(f"[app.py] User ID from session: {user_id}")
    
    nav_login = []

    st.markdown("")
    st.markdown("")

    # Get user profile using the function from auth_functions
    user_profile = get_user_profile(user_id)
    print(f"[app.py] User profile retrieved: {user_profile is not None}")

    # Safe user name handling
    user_name = "User"  # Default value
    
    if user_profile:
        # Safely get user name with multiple fallbacks
        user_name = user_profile.get("Name") 
        if not user_name and user_profile.get("email"):
            user_name = user_profile.get("email", "").split("@")[0]
        user_name = user_name or "User"
    else:
        print(f"[app.py] Warning: No user profile found for user_id {user_id}, creating new profile in Firestore...")
        # Create a default profile if not exists
        user_email = st.session_state.user_info.get("email", "unknown@user.com")
        profile_created = create_user_profile_in_firestore(user_id, user_email)
        
        if profile_created:
            # Get the newly created profile
            user_profile = get_user_profile(user_id)
            if user_profile:
                user_name = user_profile.get("Name") or user_profile.get("email", "").split("@")[0] or "User"
            else:
                user_name = user_email.split("@")[0] if user_email else "User"
        else:
            user_name = user_email.split("@")[0] if user_email else "User"
            st.error("Failed to create user profile. Some features may not work properly.")
    
    # Store user_profile in session state for other pages to use
    st.session_state.user_profile = user_profile
    print(f"[app.py] Final user_name: {user_name}")


    with st.sidebar:
        st.markdown(f"<h4 style='text-align: center;'>Welcome, {user_name}! 🎉</h4>", unsafe_allow_html=True)
        nav_login = st.navigation(
            [
                st.Page("user_pages/dashboard.py", title="Dashboard", default=True),
                st.Page("user_pages/advisor.py", title="Finanace Advisor"),
                st.Page("user_pages/lessons.py", title=" Finance Lessons"),
                st.Page("user_pages/finance_toolkit.py", title="Financial Tools"),
                st.Page("user_pages/quiz.py", title="Quiz"),
                st.Page("user_pages/news.py", title="News"),
                st.Page("user_pages/dictionary.py", title="Dictionary"),
                st.Page("user_pages/chatbot.py", title="Chatbot"),
                st.Page("user_pages/savings_tracker.py", title="Savings"),
                st.Page("user_pages/stock_analysis.py", title="Stock Analysis"),
                st.Page("user_pages/discussion_forum.py", title = "Discussion Forum"),
                st.Page("user_options/profile_entry.py", title="Profile")
            ]
        )

        # Add buttons at the bottom
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Sign Out", key="sign_out_button"):
                sign_out()  # Trigger the sign_out function when the button is pressed
        with col2:
            if st.button("Rate Us", key="rate_us_button"):
                # rate_us_button()
                if "rate" not in st.session_state:
                    rate_us_button()  # Trigger the sign_out function when the button is pressed
                else:
                    f"You gave us {st.session_state.rate['item']} stars!"

    nav_login.run()

else:
    st.markdown("")
    st.markdown("")
    st.markdown("")

    cols = st.columns([2,1])

    with cols[0]:
        st.image("assets/homepage_image.png", use_container_width=True)

    with cols[1]:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("<h1 style='text-align: center; font-size: 50px; font-weight: bold;'>WELCOME TO MONEYMENTOR</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <p>Discover tools, resources, and advice to make informed financial decisions.</p>
            <p>Explore personalized financial advice, track savings, analyse stocks, and more.</p>
        </div>
        """, unsafe_allow_html=True)

    # initialize_firebase()
    
    nav = st.navigation(
        [
            st.Page("user_pages/dashboard.py",title="Introduction", default=True),  # Magic works
            st.Page("user_options/login_pg.py", title="Log In"),  # Magic does not work
            st.Page("user_options/signup_pg.py", title="Sign Up"),  # Magic works
            st.Page("user_options/forgot_password_pg.py", title="Reset Password"),  # Magic works
        ]
    )
    nav.run()
