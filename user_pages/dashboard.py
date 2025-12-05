import streamlit as st
import plotly.express as px
from auth_functions import *
from ChatBot.chatbot import *
import streamlit_antd_components as sac
from streamlit_extras.stylable_container import stylable_container
from user_pages.contact import show_contact_form
from user_pages.disclaimer import show_disclaimer

css_styles="""
    button {
        background-color: rgba(131, 158, 101, 0.8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
        font-size: 16px;
    }
    button:hover {
        background-color: #839E65;
        color: white;
    }
    
    .floating-box-container {
        background-color: #f9f9f9; /* Light gray background for the container */
        border-radius: 20px;      /* Rounded corners */
        padding: 30px;            /* Padding inside the container */
        margin: 40px 0;           /* Margin to separate from other content */
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); /* Soft shadow effect */
    }

    .floating-box {
        position: relative;
        background-color: #f5f3eb; /* White background for the box */
        border-radius: 15px;       /* Rounded corners */
        padding: 30px;             /* Padding inside the box */
        display: flex;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); /* Box shadow for depth */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 370px
    }

    .floating-box:hover {
        transform: translateY(-10px); /* Hover effect - lift the box */
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2); /* Enhanced shadow on hover */
    }

    .floating-box img {
        border-radius: 10px;
        max-width: 100%;  /* Ensure image fits well */
        height: auto;     /* Maintain aspect ratio */
        max-height: 250px; /* Optional: limit image size */
        margin-right: 20px; /* Space between image and content */
    }

    .floating-box .content {
        padding-left: 20px;
        flex: 1;
        min-width: 250px;   /* Minimum width for the content area */
    }

    .floating-box .content h3 {
        margin-top: 0;
        font-size: 24px;
        color: #333;
    }

    .floating-box .content p {
        font-size: 15px;
        color: #555;
    }

    .floating-box .content ul {
        padding-left: 20px;
        font-size: 15px;
        color: #555;
    }

    .floating-box .content p.bold-text {
        font-weight: bold;
        font-size: 16px;
        color: #333;
    }

    @media (max-width: 768px) {
        .floating-box {
            flex-direction: column;
            align-items: center;
        }

        .floating-box img {
            margin-right: 0;
            margin-bottom: 20px; /* Add space between image and content on small screens */
        }

        .floating-box .content {
            padding-left: 0;
            text-align: center;
        }
    }

    .card-container {
        display: flex;
        justify-content: space-between; /* equal spacing between cards */
        align-items: center;
        gap: 20px;
        padding: 20px;
        margin: auto;
        flex-wrap: wrap; /* wrap to next line on small screens */
    }

    .card{
        background: linear-gradient(135deg, #dbd5bd, #f5f3eb);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
        margin: auto;
    }

    .card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        border-radius: 20px;
    }

    .card-icon {
        font-size: 70px;
        margin-bottom: 3px;
    }
    .card-title {
        font-size: 25px;
        font-weight: bold;
        margin-bottom: 7px;
        color: #333;
    }
    .card-description {
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
    }
    .card button {
        background: #839E65;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 16px;
        cursor: pointer;
        transition: background 0.3s;
    }
    .card button:hover {
        background: #6F8A50;
        color: white;
    }
"""


features = [
    # ("📊", "Financial Advisor", "Get personalized financial advice."),
    # ("📚", "Lessons", "Enhance your financial knowledge."),
    # ("🛡️", "Finance Toolkit", "Your toolkit for smart decisions."),
    # ("📝", "Finance Quiz", "Test your financial knowledge."),
    ("📰", "Finance News", "Stay updated on financial news."),
    # ("📖", "Finance Dictionary", "Easily look up financial terms."),
    ("🤖", "AI Chatbot", "Chat with our AI financial assistant."),
    ("💰", "Stock Analysis", "Monitor and analyze your savings."),
]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOGIN DASHBOARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if 'user_info' not in st.session_state:
    
    # Feature Widgets
    columns = st.columns((1,1,1))

    with columns[1]: 
        with stylable_container("blue_button", css_styles):
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            if st.button(" 🚀 Get Started", use_container_width=True, type='primary'):
                st.switch_page("user_options/login_pg.py")

    cols = st.columns(3)

    
    with stylable_container("card_container", css_styles):
        
        # Loop through features and display each card
        for i, (icon, title, description) in enumerate(features):
            with cols[i % 4]:
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown(f"""
                <div class="card">
                    <div class="card-icon">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-description">{description}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("")

    st.markdown("")
    st.markdown("")    
    st.warning("Please log in to access the dashboard.")
    st.stop()  



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN DASHBOARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
        <p>Discover tools, resources, and advice to make informed financial decisions. Explore personalized financial advice, track savings, detect fraud schemes, and much more.</p>
    </div>
    """, unsafe_allow_html=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Search Bar for AI Chatbot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown("") 
st.markdown("") 
st.markdown("")
st.markdown("")

user_input = st.text_input("🔍 Ask us anything:", placeholder="What is the best long term investment...")

cols = st.columns([1,5])
# Store user input in session state
with cols[0]:
    if st.button("Search"):
        if user_input:  # Ensure input is not empty
            st.session_state.chatbot = FinancialChatBot()
            st.session_state.history = []

            bot_response = st.session_state.chatbot.chat(user_input, None)

            st.session_state.history.append({
                "role": "user",
                "content": user_input,
                "image_path": None
            })
            st.session_state.history.append({
                "role": "assistant",
                "text": bot_response["text"],
                "plot": bot_response["plot"]
            })

            st.switch_page("user_pages/chatbot.py") 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Features Grid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown("") 
st.markdown("<h2 style='text-align: center; font-size: 40px; '>Explore Our Features</h2>", unsafe_allow_html=True)
st.markdown("") 
st.markdown("") 


features_with_links = [
    # ("📊", "Financial Advisor", "Get personalized financial advice.", "advisor"),
    # ("📚", "Lessons", "Enhance your financial knowledge.", "lessons"),
    # ("🛡️", "Finance Toolkit", "Your toolkit for smart decisions.", "finance_toolkit"),
    # ("📝", "Finance Quiz", "Test your financial knowledge.", "quiz"),
    ("📰", "Finance News", "Stay updated on financial news.", "news"),
    # ("📖", "Finance Dictionary", "Easily look up financial terms.", "dictionary"),
    ("🤖", "AI Chatbot", "Chat with our AI financial assistant.", "chatbot"),
    ("💰", "Stock Analysis", "Monitor and analyze your savings.", "stock_analysis"),
]


# Apply styling with Streamlit Extras
with stylable_container("card_container", css_styles):
    cols = st.columns(3)
    for i, (icon, title, description, page) in enumerate(features_with_links):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            if st.button(f"Open", key=f"button_{i}"):
                st.switch_page(f"user_pages/{page}.py")
                st.rerun()
            # Use page switch functionality for navigation
    
            st.markdown("") 
            st.markdown("") 
            
        
            
            





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Floating Box with Image and Text ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown("") 
st.markdown("") 
st.markdown("") 


with stylable_container("floating-box-container", css_styles):
    
    col1, col2 = st.columns([1, 1])  
    
    with col1:
        st.image("assets/output-onlinepngtools.png", use_container_width=True, caption="Financial Education")
    
    with col2:
        st.markdown("""
        <div class="floating-box">
            <div class="content">
                <h3 style="margin-top: 0;">Your Path to Financial Freedom Starts Here!</h3>
                <p style="font-size: 15px;">Did you know that 76% of Indians struggle with understanding basic financial concepts? You're not alone—but there's a solution! Our app is designed to make mastering your finances easy, fun, and empowering with:</p>
                <ul style="padding-left: 20px; font-size: 15px;">
                    <li style="margin-bottom: 8px;">Bite-sized lessons that are easy to grasp and stick with you.</li>
                    <li style="margin-bottom: 8px;">AI-powered tools tailored to your unique financial goals and needs.</li>
                    <li style="margin-bottom: 8px;">Real-world skills to confidently grow your wealth and secure your future.</li>
                </ul>
                <p style="font-size: 15px;">We're closing the financial literacy gap—one user at a time. Your journey to smart money habits starts now! 💡</p>
                <p style="font-weight: bold; font-size: 15px;">Join <span style="color: rgba(131, 158, 110);">smart</span> Indians already taking control of their finances!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FAQ Section ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

st.markdown("") 
st.markdown("") 
st.markdown("") 

faq_col1, faq_col2 = st.columns(2)

with faq_col1:
    st.markdown("<h2>Frequently asked questions</h2>", unsafe_allow_html=True)
    st.markdown("Can't find what you're looking for? We are always happy to help you navigate your financial journey!")

with faq_col2:
    with st.expander("What is Financial Planning?"):
        st.write("Financial planning is the process of managing your money to achieve your short- and long-term financial goals. It includes budgeting, saving, investing, insurance, and planning for future needs like education or retirement. It helps you understand your current financial situation and create a strategy to improve it. Overall, financial planning ensures you use your resources wisely and stay financially secure.")
    with st.expander("How to start saving money?"):
        st.write("Start by tracking your income and expenses so you know where your money is going. Create a simple budget and set a small, realistic savings goal. Pay yourself first by automatically transferring a fixed amount into a savings account each month. Cut unnecessary expenses, like subscriptions you don’t use or frequent takeout. Gradually increase your savings as your habits improve.")
    with st.expander("What are the basics of investing?"):
        st.write("The basics of investing start with understanding your financial goals and how much risk you can handle. You then choose where to invest—common options include stocks, bonds, mutual funds, and index funds. Diversification is key, meaning you spread your money across different assets to reduce risk. It’s also important to invest for the long term and stay consistent. Finally, learn how returns, risk, and time work together so you can make informed decisions.")
    with st.expander("How to detect a fraud scheme?"):
        st.write("You can detect a fraud scheme by looking for red flags like promises of guaranteed or unusually high returns with little or no risk. Be cautious if you’re pressured to act quickly or if details are unclear or secretive. Verify the company’s licenses, reviews, and contact information from trusted sources. Avoid sharing personal or financial details unless you’re sure the organization is legitimate. Always trust your instincts— if something feels off, investigate further or walk away.")





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FOOTER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


st.markdown("<hr>", unsafe_allow_html=True)
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown("<h2>Join the MoneyMentor club today!</h2>", unsafe_allow_html=True)

with footer_col2:
    st.markdown("") 
    
    sac.segmented(
        items=[
            sac.SegmentedItem(label='github', icon='github',href="https://github.com/JJFIREE/moneyMentor"),
            sac.SegmentedItem(label='mail', icon='google',href="mailto:jai.aggarwal.ug22@nsut.ac.in"),
        ], index=3, align='right',size='lg', color='rgba(131, 158, 101, 0.8)'
    )



@st.dialog("📨 Contact Us",width="large")
def contact():
        show_contact_form()

@st.dialog("⚠️ Disclaimer",width="large")
def disclaimer():
    show_disclaimer()

# footer_col3, footer_col4 = st.columns(2)
# with footer_col3:
#     col1, col2 = st.columns([1,1])

#     # with col1:
#     #     if st.button("Contact Us", key="contact_button"):
#     #         contact()
    
#     with col2:
if st.button("Disclaimer", key="disclaimer_button"):
    disclaimer()