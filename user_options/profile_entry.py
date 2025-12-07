import streamlit as st
import auth_functions
from auth_functions import *
from datetime import datetime
from firebase_admin import credentials, firestore

# Custom CSS for styling
st.markdown("""
<style>
    .profile-header h1 {
        color: #556b3b;
        font-size: 60px;
    }
    .profile-card {
        background-color: rgba(165, 199, 127, 0.4);
        border-radius: 10px;
        border: 0.7px solid #76b5c5;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f5f3eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid rgba(131, 158, 101, 0.8);
    }
    .risk-high {
        color: #E74C3C;
        font-weight: bold;
    }
    .risk-medium {
        color: #F39C12;
        font-weight: bold;
    }
    .risk-low {
        color: #27AE60;
        font-weight: bold;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

def update_user_profile_in_userdata(user_id, profile_data):
    """Update the user's profile information in UserData collection"""
    try:
        # Initialize Firebase if needed
        initialize_firebase_once()
        
        # Get Firestore client
        db = firestore.client()
        
        # Update in UserData collection (for profile_entry.py)
        doc_ref = db.collection("UserData").document(user_id)
        doc_ref.set(profile_data, merge=True)  # Use merge=True to update fields without overwriting
        
        # Also update in UserProfiles collection for consistency
        user_prof_ref = db.collection("UserProfiles").document(user_id)
        user_prof_ref.set(profile_data, merge=True)
        
        # Also update in users collection for compatibility
        users_ref = db.collection("users").document(user_id)
        users_ref.set(profile_data, merge=True)
        
        st.success("Profile updated successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error updating profile: {e}")
        return False

def calculate_risk_profile(income, investing_experience, savings, investment_pref):
    """Safely calculate risk profile with default values"""
    try:
        income = float(income) if income else 0
        investing_experience = float(investing_experience) if investing_experience else 0
        savings = float(savings) if savings else 0
        
        savings_ratio = savings / income if income > 0 else 0
        risk_score = (investing_experience * 2) + (savings_ratio * 5)

        if investment_pref == "Aggressive":
            risk_score += 5
        elif investment_pref == "Moderate":
            risk_score += 2

        if risk_score > 12:
            return "High Risk"
        elif risk_score > 6:
            return "Medium Risk"
        else:
            return "Low Risk"
    except:
        return "Medium Risk"  # Default if calculation fails

# Main Profile Page
st.markdown("""
<div class="profile-header">
    <h1 style="text-align:center;">👤 Your Profile</h1>
    <p style="text-align:center;">Manage your personal and financial information</p>
</div>
""", unsafe_allow_html=True)
st.markdown("")
st.markdown("")

if "user_info" in st.session_state:
    user_id = st.session_state.user_id
    
    # Use the get_user_profile from auth_functions
    user_profile = get_user_profile(user_id)

    if user_profile:
        # Safe access to user profile data with defaults
        name = user_profile.get("Name", "New User")
        age = user_profile.get("Age", 18)
        income = user_profile.get("Income", 0)
        investing_experience = user_profile.get("Investing Experience", 0)
        savings = user_profile.get("Savings", 0)
        investment_pref = user_profile.get("Investment Preferences", "Conservative")
        financial_goal = user_profile.get("Financial Goal", "")
        risk_tolerance = user_profile.get("Risk Tolerance", "Medium")
        investment_types = user_profile.get("Investment Types", [])
        
        # Display Profile Information
        st.markdown("### 📋 Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Name</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(name), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Age</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(age), unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Investing Experience</div>
                <div class="metric-value">{} years</div>
            </div>
            """.format(investing_experience), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Income</div>
                <div class="metric-value">${:,.2f}</div>
            </div>
            """.format(income), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Savings</div>
                <div class="metric-value">${:,.2f}</div>
            </div>
            """.format(savings), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Investment Preferences</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(investment_pref), unsafe_allow_html=True)
        
        # Risk Profile Section
        risk_category = calculate_risk_profile(income, investing_experience, savings, investment_pref)
        
        risk_class = "risk-high" if risk_category == "High Risk" else \
                    "risk-medium" if risk_category == "Medium Risk" else "risk-low"
        
        st.markdown(f"""
        <div class="profile-card">
            <h3>📊 Your Risk Profile</h3>
            <p>Based on your financial information, your risk tolerance is:</p>
            <div style="font-size: 1.5rem; margin: 1rem 0;" class="{risk_class}">
                {risk_category}
            </div>
        """, unsafe_allow_html=True)
        
        if risk_category == "Low Risk":
            st.info("💡 You prefer safety. Consider diversified mutual funds and bonds.")
        elif risk_category == "Medium Risk":
            st.info("📈 You have a balanced approach. ETFs, blue-chip stocks, and REITs are good options.")
        else:
            st.warning("⚠️ High risk! Look into growth stocks, crypto, or real estate carefully.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        # Edit Profile Section
        st.markdown("### ✏️ Update Profile")
        with st.expander("Edit Profile Information", expanded=False):
            with st.form("profile_form"):
                name = st.text_input("Name", value=name)
                age = st.number_input("Age", value=age, step=1, min_value=1, max_value=120)
                income = st.number_input("Income", value=float(income), step=100.0, min_value=0.0)
                investing_experience = st.number_input("Investing Experience (Years)", 
                                                     value=int(investing_experience), step=1, min_value=0)
                savings = st.number_input("Savings", value=float(savings), step=100.0, min_value=0.0)
                investment_pref = st.selectbox("Investment Preferences", 
                                             ["Conservative", "Moderate", "Aggressive"], 
                                             index=["Conservative", "Moderate", "Aggressive"].index(investment_pref) if investment_pref in ["Conservative", "Moderate", "Aggressive"] else 0)
                financial_goal = st.text_input("Your Financial Goal (e.g., Save $10,000)", value=financial_goal)
                risk_tolerance = st.selectbox("Your Risk Tolerance", ["Low", "Medium", "High"], 
                                            index=["Low", "Medium", "High"].index(risk_tolerance) if risk_tolerance in ["Low", "Medium", "High"] else 1)
                investment_types = st.multiselect("Your Investment Preferences", 
                                                 ["Stocks", "Bonds", "Real Estate", "Cryptocurrency"],
                                                 default=investment_types)

                cols = st.columns([1.5,1,10])
                with cols[0]:
                    submitted = st.form_submit_button("Save Changes")
                with cols[1]:
                    if st.form_submit_button("Cancel"):
                        pass

                if submitted:
                    updated_profile = {
                        "Name": name,
                        "Age": age,
                        "Income": income,
                        "Investing Experience": investing_experience,
                        "Savings": savings,
                        "Investment Preferences": investment_pref,
                        "Financial Goal": financial_goal,
                        "Risk Tolerance": risk_tolerance,
                        "Investment Types": investment_types
                    }
                    success = update_user_profile_in_userdata(user_id, updated_profile)
                    if success:
                        # Update session state
                        st.session_state.user_profile = get_user_profile(user_id)
                        st.rerun()
        
        st.markdown("")
        st.markdown("")
        # Delete Account Section
        st.markdown("### ❌ Account Actions")
        with st.expander("Delete Account", expanded=False):
            st.warning("This action cannot be undone. All your data will be permanently deleted.")
            password = st.text_input(label='Confirm your password', type='password')
            if st.button(label='Delete Account', type='primary'):
                auth_functions.delete_account(password)
    else:
        st.error("User profile not found. Please try refreshing the page.")
else:
    st.warning("You need to log in first.")