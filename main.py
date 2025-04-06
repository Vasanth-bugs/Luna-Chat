import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
index = faiss.IndexFlatL2(384)

# Define Gemini Model
def configure_gemini(api_key):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    return None

# Streamlit UI
st.set_page_config(page_title="LUNA - Educational Chatbot", layout="wide")

# Sidebar Menu
menu = st.sidebar.radio("Navigation", ["Home", "Chatbot", "Analytics", "Settings"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY
if "model" not in st.session_state and st.session_state.api_key:
    st.session_state.model = configure_gemini(st.session_state.api_key)

# Function to generate visualizations
def generate_visualization(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['x'], data['y'], marker='o')
    plt.title('Sample Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    image_path = "visualization.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

# Function to get chatbot response
def chat_with_gemini(prompt: str):
    if not st.session_state.api_key:
        return "API key is not set. Please configure it in Settings.", None
    try:
        response = st.session_state.model.generate_content(prompt)
        response_text = response.text
        # Check if the response indicates a need for visualization
        if "visualize" in prompt.lower():
            # Example data; replace with actual data processing
            data = pd.DataFrame({
                'x': range(10),
                'y': [i**2 for i in range(10)]
            })
            image_path = generate_visualization(data)
            return response_text, image_path
        return response_text, None
    except Exception as e:
        return str(e), None

if menu == "Home":
    st.title("📚 Welcome to LUNA - Your AI Educational Assistant")
    st.write("LUNA is an advanced AI-powered chatbot designed to assist students with educational queries. It provides instant responses, helpful explanations, and insightful analytics to enhance your learning experience.")

    st.subheader("✨ About LUNA")
    st.write(
        "- Uses AI to answer academic questions in real-time.\n"
        "- Helps students with various subjects by providing quick and accurate explanations.\n"
        "- Tracks and analyzes user queries for better recommendations.\n"
        "- Designed for an interactive and engaging learning experience."
    )

elif menu == "Chatbot":
    st.title("💬 AI Chatbot")
    
    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"])

    # User input
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get bot response
        bot_response, image_path = chat_with_gemini(user_input)

        # Display bot message
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            if image_path:
                st.image(image_path)

        # Add bot response to chat history
        message = {"role": "assistant", "content": bot_response}
        if image_path:
            message["image"] = image_path
        st.session_state.chat_history.append(message)

elif menu == "Analytics":
    st.title("📊 Analytics Dashboard")
    
    if len(st.session_state.chat_history) > 0:
        # Count total messages
        total_messages = len(st.session_state.chat_history)
        user_messages = sum(1 for msg in st.session_state.chat_history if msg["role"] == "user")
        bot_messages = total_messages - user_messages
        
        st.metric("Total Messages", total_messages)
        st.metric("User Messages", user_messages)
        st.metric("Bot Messages", bot_messages)
        
        # Prepare data for visualization
        data = pd.DataFrame(
            {"Type": ["User", "Bot"], "Messages": [user_messages, bot_messages]}
        )
        
        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(data["Type"], data["Messages"], color=["blue", "green"])
        ax.set_ylabel("Message Count")
        ax.set_title("User vs Bot Messages")
        st.pyplot(fig)
    else:
        st.write("No chat data available yet. Start a conversation in the Chatbot section!")

elif menu == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("API Key Configuration")
    api_key_input = st.text_input("Enter your Gemini API Key:", st.session_state.api_key, type="password")
    if st.button("Save API Key"):
        st.session_state.api_key = api_key_input
        st.session_state.model = configure_gemini(api_key_input)
        st.success("API Key updated successfully!")
