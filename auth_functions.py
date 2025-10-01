# import json
# import requests
# import streamlit as st


# ## -------------------------------------------------------------------------------------------------
# ## Firebase Auth API -------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------

# def initialize_firebase_once():
#     """Initialize Firebase once per session"""
#     if 'firebase_initialized' not in st.session_state or not st.session_state.firebase_initialized:
#         initialize_firebase()
#         st.session_state.firebase_initialized = True



# def sign_in_with_email_and_password(email, password):
#     print(1)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8"}
#     data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def get_account_info(id_token):
#     print(2)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getAccountInfo?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8"}
#     data = json.dumps({"idToken": id_token})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def send_email_verification(id_token):
#     print(3)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8"}
#     data = json.dumps({"requestType": "VERIFY_EMAIL", "idToken": id_token})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def send_password_reset_email(email):
#     print(4)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8"}
#     data = json.dumps({"requestType": "PASSWORD_RESET", "email": email})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def create_user_with_email_and_password(email, password):
#     print(5)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8" }
#     data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def delete_user_account(id_token):
#     print(6)
#     print(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/deleteAccount?key={0}".format(st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
#     headers = {"content-type": "application/json; charset=UTF-8"}
#     data = json.dumps({"idToken": id_token})
#     request_object = requests.post(request_ref, headers=headers, data=data)
#     raise_detailed_error(request_object)
#     return request_object.json()

# def raise_detailed_error(request_object):
#     try:
#         request_object.raise_for_status()
#     except requests.exceptions.HTTPError as error:
#         raise requests.exceptions.HTTPError(error, request_object.text)

# ## -------------------------------------------------------------------------------------------------
# ## Authentication functions ------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------

# def sign_in(email:str, password:str) -> None:
#     try:
#         # Attempt to sign in with email and password
#         id_token = sign_in_with_email_and_password(email,password)['idToken']

#         # Get account information
#         user_info = get_account_info(id_token)["users"][0]

#         # If email is not verified, send verification email and do not sign in
#         if not user_info["emailVerified"]:
#             send_email_verification(id_token)
#             st.session_state.auth_warning = 'Check your email to verify your account'

#         # Save user info to session state and rerun
#         else:
#             st.session_state.user_info = user_info
#             # st.experimental_rerun()

#     except requests.exceptions.HTTPError as error:
#         error_message = json.loads(error.args[1])['error']['message']
#         if error_message in {"INVALID_EMAIL","EMAIL_NOT_FOUND","INVALID_PASSWORD","MISSING_PASSWORD"}:
#             st.session_state.auth_warning = 'Error: Use a valid email and password'
#         else:
#             st.session_state.auth_warning = 'Error: Please try again later'

#     except Exception as error:
#         print(error)
#         st.session_state.auth_warning = 'Error: Please try again later'


# def create_account(email: str, password: str) -> bool:
#     try:
#         # Create account (and save id_token)
#         response = create_user_with_email_and_password(email, password)
#         id_token = response['idToken']
#         user_id = response['localId']  # Get the user ID (uid)
#         create_user_profile_in_firestore(user_id, email) #send verification
#         send_email_verification(id_token)
#         return True
    
#     except requests.exceptions.HTTPError as error:
#         error_message = json.loads(error.args[1])['error']['message']
#         if error_message == "EMAIL_EXISTS":
#             st.session_state.auth_warning = 'Error: Email belongs to existing account'
#         elif error_message in {"INVALID_EMAIL","INVALID_PASSWORD","MISSING_PASSWORD","MISSING_EMAIL","WEAK_PASSWORD"}:
#             st.error(error_message)
#             st.session_state.auth_warning = 'Error: Use a valid email and password'
#         else:
#             st.session_state.auth_warning = 'Error: Please try again later'
    
#     except Exception as error:
#         print(error)
#         st.session_state.auth_warning = 'Error: Please try again later'

# def reset_password(email:str) -> None:
#     try:
#         send_password_reset_email(email)
#         st.session_state.auth_success = 'Password reset link sent to your email'
    
#     except requests.exceptions.HTTPError as error:
#         error_message = json.loads(error.args[1])['error']['message']
#         if error_message in {"MISSING_EMAIL","INVALID_EMAIL","EMAIL_NOT_FOUND"}:
#             st.session_state.auth_warning = 'Error: Use a valid email'
#         else:
#             st.session_state.auth_warning = 'Error: Please try again later'    
    
#     except Exception:
#         st.session_state.auth_warning = 'Error: Please try again later'


# def sign_out() -> None:
#     st.session_state.clear()
#     st.session_state.auth_success = 'You have successfully signed out'


# def delete_account(password:str) -> None:
#     try:
#         # Confirm email and password by signing in (and save id_token)
#         id_token = sign_in_with_email_and_password(st.session_state.user_info['email'],password)['idToken']
        
#         # Attempt to delete account
#         delete_user_account(id_token)
#         st.session_state.clear()
#         st.session_state.auth_success = 'You have successfully deleted your account'

#     except requests.exceptions.HTTPError as error:
#         error_message = json.loads(error.args[1])['error']['message']
#         print(error_message)

#     except Exception as error:
#         print(error)


# from google.cloud import firestore
# import firebase_admin
# from firebase_admin import credentials, firestore
# import json
# import requests
# import streamlit as st
# # import streamlit_cookie_manager as cookie_manager


# ## -------------------------------------------------------------------------------------------------
# ## Firebase and authentication -------------------------------------------------------------------------------
# ## -------------------------------------------------------------------------------------------------

# # Global variable for Firestore client
# db = None
# import toml

# def initialize_firebase():
#     """Initialize Firebase only once if not already done"""
#     global db
#     if not firebase_admin._apps:
#         # Load credentials from the TOML file
#         toml_config = toml.load(".streamlit/secrets.toml")
        
#         # Extract Firebase credentials from the TOML file
#         firebase_credentials = toml_config.get("FIREBASE")  # Assuming the TOML contains "textkey" with the JSON string
        
#         if "private_key" in firebase_credentials:
#             firebase_credentials["private_key"] = firebase_credentials["private_key"].replace("\\n", "\n")

        
#         if firebase_credentials:
#             # Convert the json string to dictionary
#             # import json
#             # firebase_credentials = json.loads(firebase_config)

#             # Use the credentials to initialize Firebase
#             cred = credentials.Certificate(firebase_credentials)
#             #cred = credentials.Certificate('firebase-key.json')
#             firebase_admin.initialize_app(cred)
#             db = firestore.client()
#         else:
#             print("No credentials found in TOML file.")
#     elif db is None:
#         db = firestore.client()


# # def initialize_firebase():
# #     global db
# #     if not firebase_admin._apps:
# #         # Get the dictionary from secrets
# #         firebase_secrets = st.secrets["FIREBASE"].copy()

# #         # Replace literal "\n" in private_key with real newlines
# #         if "private_key" in firebase_secrets:
# #             firebase_secrets["private_key"] = firebase_secrets["private_key"].replace("\\n", "\n")

# #         # Initialize Admin SDK
# #         cred = credentials.Certificate(firebase_secrets)
# #         firebase_admin.initialize_app(cred)
# #         db = firestore.client()
# #     elif db is None:
# #         db = firestore.client()





# def create_user_profile_in_firestore(user_id, email):
#     # Create a new document for the user in Firestore
    
#     initialize_firebase_once()
    
#     user_prof = db.collection('UserProfiles').document(user_id)
#     # user_data = db.collection('UserData').document(user_id)

#     # Initialize user profile with basic info
#     user_prof.set({
#         'email': email,
#         'first_login': True,
#         'current_login': False,
#         'createdAt': firestore.SERVER_TIMESTAMP
#     })
#     # user_data.set({
#     #     'Name': None,
#     #     'Investing Experienc': None,
#     #     'Income': None,
#     #     'Age': None
#     # })



import json
import requests
import streamlit as st
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore

## -------------------------------------------------------------------------------------------------
## Firebase Auth API -------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

def sign_in_with_email_and_password(email, password):
    print("[sign_in_with_email_and_password] Start")
    print("FIREBASE_WEB_API_KEY:", st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
    print("Sending POST request to Firebase sign-in API...")
    request_object = requests.post(request_ref, headers=headers, data=data)
    print("Response status:", request_object.status_code)
    raise_detailed_error(request_object)
    result = request_object.json()
    print("sign_in_with_email_and_password result:", result)
    return result

def get_account_info(id_token):
    print("[get_account_info] Start")
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getAccountInfo?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"idToken": id_token})
    print("Sending POST request to get account info...")
    request_object = requests.post(request_ref, headers=headers, data=data)
    print("Response status:", request_object.status_code)
    raise_detailed_error(request_object)
    result = request_object.json()
    print("get_account_info result:", result)
    return result

def send_email_verification(id_token):
    print("[send_email_verification] Start")
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"requestType": "VERIFY_EMAIL", "idToken": id_token})
    print("Sending email verification request...")
    request_object = requests.post(request_ref, headers=headers, data=data)
    print("Response status:", request_object.status_code)
    raise_detailed_error(request_object)
    result = request_object.json()
    print("send_email_verification result:", result)
    return result

def send_password_reset_email(email):
    print("[send_password_reset_email] Start")
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"requestType": "PASSWORD_RESET", "email": email})
    print("Sending password reset request...")
    request_object = requests.post(request_ref, headers=headers, data=data)
    print("Response status:", request_object.status_code)
    raise_detailed_error(request_object)
    result = request_object.json()
    print("send_password_reset_email result:", result)
    return result

def create_user_with_email_and_password(email, password):
    print("[create_user_with_email_and_password] Start")
    print("[create_user_with_email_and_password] API KEY:", st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})
    
    print("[create_user_with_email_and_password] Sending create user request...")
    try:
        request_object = requests.post(request_ref, headers=headers, data=data, timeout=10)
        print("[create_user_with_email_and_password] Response received")
        print("[create_user_with_email_and_password] Response status:", request_object.status_code)
        print("[create_user_with_email_and_password] Response text:", request_object.text)
        raise_detailed_error(request_object)
        return request_object.json()
    except requests.exceptions.Timeout:
        print("[create_user_with_email_and_password] ERROR: Request timed out")
    except Exception as e:
        print("[create_user_with_email_and_password] Exception:", e)


def delete_user_account(id_token):
    print("[delete_user_account] Start")
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/deleteAccount?key={0}".format(
        st.secrets["FIREBASE"]["FIREBASE_WEB_API_KEY"])
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"idToken": id_token})
    print("Sending delete user request...")
    request_object = requests.post(request_ref, headers=headers, data=data)
    print("Response status:", request_object.status_code)
    raise_detailed_error(request_object)
    result = request_object.json()
    print("delete_user_account result:", result)
    return result

def raise_detailed_error(request_object):
    try:
        request_object.raise_for_status()
    except requests.exceptions.HTTPError as error:
        print("HTTP error:", error, request_object.text)
        raise requests.exceptions.HTTPError(error, request_object.text)

## -------------------------------------------------------------------------------------------------
## Authentication functions ------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

def sign_in(email: str, password: str) -> None:
    print("[sign_in] Start")
    try:
        # Attempt to sign in with email and password
        id_token = sign_in_with_email_and_password(email, password)['idToken']
        print("[sign_in] Got id_token:", id_token)

        # Get account information
        user_info = get_account_info(id_token)["users"][0]
        print("[sign_in] user_info:", user_info)

        # If email is not verified, send verification email and do not sign in
        if not user_info["emailVerified"]:
            print("[sign_in] Email not verified, sending verification email")
            send_email_verification(id_token)
            st.session_state.auth_warning = 'Check your email to verify your account'

        # Save user info to session state
        else:
            st.session_state.user_info = user_info
            print("[sign_in] User signed in successfully")
            # st.experimental_rerun()

    except requests.exceptions.HTTPError as error:
        error_message = json.loads(error.args[1])['error']['message']
        print("[sign_in] HTTPError message:", error_message)
        if error_message in {"INVALID_EMAIL", "EMAIL_NOT_FOUND", "INVALID_PASSWORD", "MISSING_PASSWORD"}:
            st.session_state.auth_warning = 'Error: Use a valid email and password'
        else:
            st.session_state.auth_warning = 'Error: Please try again later'

    except Exception as error:
        print("[sign_in] Exception:", error)
        st.session_state.auth_warning = 'Error: Please try again later'


def create_account(email: str, password: str) -> bool:
    print("[create_account] Start")
    try:
        # Create account (and save id_token)
        response = create_user_with_email_and_password(email, password)
        print("[create_account] create_user response:", response)
        id_token = response['idToken']
        user_id = response['localId']  # Get the user ID (uid)
        print("[create_account] user_id:", user_id)

        # Create user profile in Firestore
        create_user_profile_in_firestore(user_id, email)
        print("[create_account] Firestore profile created")

        # Send email verification
        send_email_verification(id_token)
        print("[create_account] Verification email sent")
        return True

    except requests.exceptions.HTTPError as error:
        error_message = json.loads(error.args[1])['error']['message']
        print("[create_account] HTTPError message:", error_message)
        if error_message == "EMAIL_EXISTS":
            st.session_state.auth_warning = 'Error: Email belongs to existing account'
        elif error_message in {"INVALID_EMAIL", "INVALID_PASSWORD", "MISSING_PASSWORD", "MISSING_EMAIL", "WEAK_PASSWORD"}:
            st.error(error_message)
            st.session_state.auth_warning = 'Error: Use a valid email and password'
        else:
            st.session_state.auth_warning = 'Error: Please try again later'

    except Exception as error:
        print("[create_account] Exception:", error)
        st.session_state.auth_warning = 'Error: Please try again later'


def reset_password(email: str) -> None:
    print("[reset_password] Start")
    try:
        send_password_reset_email(email)
        print("[reset_password] Reset email sent")
        st.session_state.auth_success = 'Password reset link sent to your email'

    except requests.exceptions.HTTPError as error:
        error_message = json.loads(error.args[1])['error']['message']
        print("[reset_password] HTTPError message:", error_message)
        if error_message in {"MISSING_EMAIL", "INVALID_EMAIL", "EMAIL_NOT_FOUND"}:
            st.session_state.auth_warning = 'Error: Use a valid email'
        else:
            st.session_state.auth_warning = 'Error: Please try again later'

    except Exception as error:
        print("[reset_password] Exception:", error)
        st.session_state.auth_warning = 'Error: Please try again later'


def sign_out() -> None:
    print("[sign_out] Clearing session")
    st.session_state.clear()
    st.session_state.auth_success = 'You have successfully signed out'


def delete_account(password: str) -> None:
    print("[delete_account] Start")
    try:
        # Confirm email and password by signing in (and save id_token)
        id_token = sign_in_with_email_and_password(st.session_state.user_info['email'], password)['idToken']
        print("[delete_account] Got id_token:", id_token)

        # Attempt to delete account
        delete_user_account(id_token)
        print("[delete_account] Account deleted successfully")
        st.session_state.clear()
        st.session_state.auth_success = 'You have successfully deleted your account'

    except requests.exceptions.HTTPError as error:
        error_message = json.loads(error.args[1])['error']['message']
        print("[delete_account] HTTPError message:", error_message)

    except Exception as error:
        print("[delete_account] Exception:", error)

## -------------------------------------------------------------------------------------------------
## Firebase and Firestore initialization -----------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# Global variable for Firestore client
db = None
import toml

def initialize_firebase():
    """Initialize Firebase only once if not already done"""
    global db
    print("[initialize_firebase] Start")
    
    if not firebase_admin._apps:
        print("[initialize_firebase] No existing Firebase apps, initializing...")
        try:
            # Load secrets.toml
            toml_config = toml.load(".streamlit/secrets.toml")
            print("[initialize_firebase] Loaded secrets.toml")

            # Get Firebase credentials dict
            firebase_credentials = toml_config.get("FIREBASE")
            if not firebase_credentials:
                raise ValueError("No 'FIREBASE' section found in secrets.toml")
            print("[initialize_firebase] Firebase credentials loaded")

            # Fix private_key newlines
            if "private_key" in firebase_credentials:
                firebase_credentials["private_key"] = firebase_credentials["private_key"].replace("\\n", "\n")
                print("[initialize_firebase] Fixed private_key newlines")

            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("[initialize_firebase] Firebase initialized successfully")

        except Exception as e:
            print("[initialize_firebase] Exception during initialization:", e)
            db = None
    elif db is None:
        print("[initialize_firebase] Firebase already initialized but db is None, creating client...")
        db = firestore.client()
        print("[initialize_firebase] Firestore client assigned")

    else:
        print("[initialize_firebase] Firebase already initialized and db exists")


def initialize_firebase_once():
    """Initialize Firebase once per session"""
    if 'firebase_initialized' not in st.session_state or not st.session_state.firebase_initialized:
        print("[initialize_firebase_once] Initializing Firebase for the session")
        initialize_firebase()
        st.session_state.firebase_initialized = True
    else:
        print("[initialize_firebase_once] Firebase already initialized for session")

from google.api_core.exceptions import DeadlineExceeded

def create_user_profile_in_firestore(user_id, email):
    try:
        print(f"[create_user_profile_in_firestore] Start, user_id: {user_id}")
        user_prof = db.collection('UserProfiles').document(user_id)
        print("[create_user_profile_in_firestore] Document reference created")
        
        # Increase timeout to 30 seconds
        user_prof.set({
            'email': email,
            'first_login': True,
            'current_login': False,
            'createdAt': firestore.SERVER_TIMESTAMP
        }, timeout=30)
        
        print("[create_user_profile_in_firestore] Firestore profile written successfully")
        
    except DeadlineExceeded:
        print("[create_user_profile_in_firestore] Warning: Firestore write timed out, but account creation succeeded")
    except Exception as e:
        print(f"[create_user_profile_in_firestore] Exception: {e}")
