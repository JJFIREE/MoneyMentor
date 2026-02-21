import streamlit as st
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

# # Load API key from .env
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

api_key = st.secrets['GOOGLE']['GEMINI_API_KEY']
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API key not found. Please set GEMINI_API_KEY in your .env file.")





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ QUIZ QUESTIONS ~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_quiz_questions(history_json, mode="general", difficulty="easy"):
    """
    Generates multiple-choice quiz questions using Gemini API.

    :param history_json: JSON string of user quiz/chat/dictionary history
    :param mode: "general" for beginner questions, "personalized" for tailored ones
    :param difficulty: "easy", "medium", or "hard"
    :return: List of multiple-choice questions
    """
    if mode == "general":
        prompt = (
            f"Generate exactly 5 {difficulty}-level finance multiple-choice questions. "
            "Follow this EXACT format for each question:\n\n"
            "Question: <question_text>\n"
            "A) <option1>\n"
            "B) <option2>\n"
            "C) <option3>\n"
            "D) <option4>\n"
            "Answer: <correct_option_letter>\n"
            "Explanation: <brief_explanation>\n\n"
            "Separate each question with a blank line."
        )
    else:
        prompt = (
            f"Generate exactly 5 {difficulty}-level personalized finance questions about budgeting, savings, and personal money management. "
            "Follow this EXACT format for each question:\n\n"
            "Question: <question_text>\n"
            "A) <option1>\n"
            "B) <option2>\n"
            "C) <option3>\n"
            "D) <option4>\n"
            "Answer: <correct_option_letter>\n"
            "Explanation: <brief_explanation>\n\n"
            "Separate each question with a blank line."
        )

    model = genai.GenerativeModel(
        "gemini-3-flash-preview",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1500,  # Reduced for faster generation
        }
    )

    try:
        response = model.generate_content(prompt)

        if not response or not response.candidates:
            return []

        raw_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""

        # Splitting response into structured questions
        questions_list = raw_text.strip().split("\n\n")
        questions = []

        for q_text in questions_list:
            lines = [line.strip() for line in q_text.strip().split("\n") if line.strip()]
            if len(lines) < 6:
                continue  # Skip malformed questions

            # Extract question (remove "Question:" prefix if present)
            question_text = lines[0]
            if question_text.startswith("Question:"):
                question_text = question_text.replace("Question:", "").strip()
            
            # Extract only the options (lines starting with A), B), C), D))
            options = []
            for i in range(1, min(5, len(lines))):
                if lines[i] and lines[i][0].upper() in ['A', 'B', 'C', 'D'] and ')' in lines[i]:
                    options.append(lines[i].strip())
            
            if len(options) != 4:
                continue  # Skip if we don't have exactly 4 options
            
            # Extract answer
            answer_line = next((line for line in lines if line.startswith("Answer:")), None)
            if not answer_line:
                continue
            correct_answer = answer_line.replace("Answer:", "").strip()[0].upper()
            
            # Extract explanation
            explanation_line = next((line for line in lines if line.startswith("Explanation:")), None)
            explanation = explanation_line.replace("Explanation:", "").strip() if explanation_line else "No explanation provided."

            questions.append({
                "question": question_text,
                "options": options,
                "answer": correct_answer,
                "explanation": explanation
            })

        return questions[:5]  # Limit to 5 questions

    except Exception as e:
        st.error(f"Error fetching quiz questions: {str(e)}")
        return []




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ STREAMLIT APP ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


st.markdown("""
    <style>
    .profile-header h1 {
        color: #556b3b;
        font-size: 60px;
    }
    </style>        
""", unsafe_allow_html = True)


st.markdown("""
<div class="profile-header">
    <h1 style="text-align:center;">📊 Finance Quiz App</h1>
    <p style="text-align:center;">Test and expand your money knowledge with interactive quizzes! Choose between general financial concepts or personalized quizzes based on your profile. 
            Adjust the difficulty to match your expertise level - perfect for beginners and experts alike. Each quiz helps you spot knowledge gaps while making finance fun!</p>
</div>
""", unsafe_allow_html=True)


st.markdown("")
st.markdown("")
# Initialize session state
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = "general"
if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = "easy"
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dictionary_searches" not in st.session_state:
    st.session_state.dictionary_searches = []
if "score" not in st.session_state:
    st.session_state.score = 0

# Quiz Settings
st.markdown("### Quiz Settings")
col1, col2 = st.columns(2)
with col1:
    quiz_mode = st.radio("Choose Quiz Type:", ["General", "Personalized"], key="quiz_type_radio")
with col2:
    difficulty = st.selectbox("Select Difficulty:", ["Easy", "Medium", "Hard"], key="difficulty_select")

# Only fetch new questions when explicitly requested or settings change
settings_changed = (
    st.session_state.quiz_mode != quiz_mode.lower() or 
    st.session_state.quiz_difficulty != difficulty.lower()
)

if st.button("Start New Quiz") or (not st.session_state.questions and settings_changed):
    with st.spinner("Generating quiz questions..."):
        # No need to pass full history for general mode
        history_json = "{}" if quiz_mode.lower() == "general" else json.dumps({
            "quiz_answers": st.session_state.quiz_history[-10:],  # Only last 10 for speed
            "finance_gpt_queries": st.session_state.chat_history[-5:],  # Only last 5
            "searched_terms": st.session_state.dictionary_searches[-10:]  # Only last 10
        }, indent=2)
        
        st.session_state.questions = get_quiz_questions(
            history_json, 
            mode=quiz_mode.lower(), 
            difficulty=difficulty.lower()
        )
        st.session_state.quiz_mode = quiz_mode.lower()
        st.session_state.quiz_difficulty = difficulty.lower()
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.rerun()

# Display questions
if st.session_state.questions:
    q_idx = st.session_state.current_question
    question_data = st.session_state.questions[q_idx]

    st.markdown("---")
    st.markdown(f"### Question {q_idx + 1} of 5")
    st.markdown(f"**{question_data['question']}**")

    # Progress bar
    progress = (q_idx + 1) / 5
    st.progress(progress)

    selected_option = st.radio("Choose your answer:", question_data["options"], key=q_idx)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Answer"):
            correct = selected_option.startswith(question_data["answer"])
            #print(correct)
            if correct:
                st.success("Correct! 🎉")
                st.session_state.score += 1
            else:
                # Find the correct option safely
                correct_option = None
                for opt in question_data["options"]:
                    if opt.startswith(question_data["answer"]):
                        correct_option = opt
                        break
                
                if correct_option:
                    st.error(f"Incorrect! The correct answer is: **{correct_option}**.\n\n**Explanation:** {question_data['explanation']}")
                else:
                    st.error(f"Incorrect! The correct answer is: **{question_data['answer']}**.\n\n**Explanation:** {question_data['explanation']}")

            # Store answer in history only for personalized mode
            if st.session_state.quiz_mode == "personalized":
                st.session_state.quiz_history.append({
                    "question": question_data["question"],
                    "selected": selected_option,
                    "correct": correct
                })

    with col2:
        if st.button("Next Question"):
            if q_idx + 1 < len(st.session_state.questions):
                st.session_state.current_question += 1
                st.rerun()
            else:
                st.success(f"Quiz completed! 🎯 Your final score is **{st.session_state.score}/5**.")
                st.session_state.questions = []  # Reset questions for a new quiz
                st.balloons()
else:
    st.info("👆 Click 'Start New Quiz' to begin!")
