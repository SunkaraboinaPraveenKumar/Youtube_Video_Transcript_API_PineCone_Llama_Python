import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from fpdf import FPDF
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="LLAMA 3.1 CHAT",
    page_icon="🦙",
    layout="centered"
)

# Load API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the LLAMA model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize session state for chat and video transcript
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# Streamlit page title
st.title("🦙 LLAMA 3.1 Video & Chat Assistant")

# Video Transcript Section
st.header("🎥 YouTube Video Analysis")
video_url = st.text_input("Enter YouTube Video URL", "")

if st.button("Process Video"):
    if video_url:
        try:
            # Extract transcript
            st.write("Fetching video transcript...")
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            docs = loader.load()
            transcript = docs[0].page_content
            st.session_state.transcript = transcript

            st.success("Video transcript fetched successfully!")
            st.text_area("Video Transcript", transcript, height=300)

        except Exception as e:
            st.error(f"Failed to process video: {e}")
    else:
        st.warning("Please enter a valid YouTube URL.")

# Summarize Section
if st.session_state.transcript:
    if st.button("Summarize Transcript"):
        try:
            # Summarize transcript
            st.write("Summarizing the transcript...")
            summarize_prompt = PromptTemplate.from_template("""
            You are a helpful assistant that summarizes transcripts:
            {context}
            Summarize the context.
            """)

            # Chain for summarization
            summary_chain = summarize_prompt | llm | RunnableLambda(lambda message: message.content.strip())

            # Invoke the chain
            summary = summary_chain.invoke({"context": st.session_state.transcript})
            st.session_state.summary = summary

            # Save summary as PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Video Transcript:\n{st.session_state.transcript}\n\nSummary:\n{summary}")
            pdf_output_path = "video_summary.pdf"
            pdf.output(pdf_output_path)

            st.success("Summary generated successfully!")
            st.download_button("Download Summary PDF", data=open(pdf_output_path, "rb"), file_name="summary.pdf")

            st.text_area("Summary", summary, height=150)

        except Exception as e:
            st.error(f"Failed to summarize transcript: {e}")

# Chat Section
st.header("💬 Chat with LLAMA")

user_prompt = st.chat_input("Ask LLAMA...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Add context from the video transcript if available
    context = f"Relevant Transcript Context:\n{st.session_state.transcript[:500]}" if st.session_state.transcript else ""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context + user_prompt}
    ]

    response = llm.invoke(messages)
    assistant_response = response.content

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").markdown(assistant_response)
