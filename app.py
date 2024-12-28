import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from fpdf import FPDF
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="🦙 LLAMA 3.1 Video & Chat Assistant with Pinecone",
    page_icon="🦙",
    layout="centered"
)

# Load API keys from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize LLAMA model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "webbot"
index = pc.Index(index_name)
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# Streamlit UI
st.title("🦙 LLAMA 3.1 Video & Chat Assistant with Pinecone")

# YouTube Video Section
st.header("🎥 YouTube Video Analysis")
video_url = st.text_input("Enter YouTube Video URL", "")

if st.button("Process Video"):
    if video_url:
        try:
            st.write("Fetching video transcript...")
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            docs = loader.load()
            if docs:
                transcript = docs[0].page_content
                st.session_state.transcript = transcript
                st.success("Video transcript fetched successfully!")
                st.text_area("Video Transcript", transcript, height=300)
            else:
                st.error("No transcript found for the video.")
        except Exception as e:
            st.error(f"Failed to process video: {e}")
    else:
        st.warning("Please enter a valid YouTube URL.")


# Function to split transcript into smaller chunks
def split_transcript(transcript_, chunk_size=1000):
    return [transcript[i:i + chunk_size] for i in range(0, len(transcript_), chunk_size)]


# Store Transcript in Vector DB Section
if st.session_state.transcript and st.button("Store Transcript in Vector DB"):
    try:
        st.write("Storing transcript in Pinecone...")

        # Split the transcript into smaller chunks to fit within Pinecone's metadata limit
        transcript_chunks = split_transcript(st.session_state.transcript)

        # Embed and store each chunk separately
        for i, chunk in enumerate(transcript_chunks):
            vectors = [{
                "id": f"transcript_chunk_{i}",
                "values": HuggingFaceEmbeddings().embed_documents([chunk])[0],
                "metadata": {"chunk": chunk}  # Only store the chunk, not the full transcript
            }]
            index.upsert(vectors)

        st.success("Transcript chunks stored in Pinecone successfully!")

    except Exception as e:
        st.error(f"Failed to store transcript: {e}")


# Fetch relevant content from Pinecone
def retrieve_relevant_docs(query_input_p, k=3):
    query_embedding = HuggingFaceEmbeddings().embed_query(query_input_p)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True
    )
    return [result['metadata']['chunk'] for result in results['matches']] if results['matches'] else [
        "No relevant context found."]


# Summarize Section
if st.session_state.transcript and st.button("Summarize Transcript"):
    try:
        st.write("Summarizing the transcript...")

        # Define Prompt Template for summarization
        summarize_prompt = PromptTemplate.from_template("""
        You are a helpful assistant that summarizes transcripts:
        {context}
        Summarize the context in simple and clear language with examples.
        """)

        context = st.session_state.transcript

        # If transcript exceeds the LLAMA context window, summarize in chunks
        if len(context) > 10000:
            st.write("Transcript exceeds context window, summarizing in chunks...")

            transcript_chunks = split_transcript(context, 10000)
            summaries = []

            for chunk in transcript_chunks:
                # Directly summarize the chunk without additional context retrieval
                summary_chain = summarize_prompt | llm | RunnableLambda(lambda message: message.content.strip())
                chunk_summary = summary_chain.invoke({"context": chunk})
                summaries.append(chunk_summary)

            # Join all chunk summaries into a final summary
            summary = " ".join(summaries)
        else:
            # LLM chain for summarization
            summary_chain = summarize_prompt | llm | RunnableLambda(lambda message: message.content.strip())
            summary = summary_chain.invoke({"context": context})

        st.session_state.summary = summary

        # Save the summary as a PDF
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

    # Retrieve relevant docs from Pinecone and use as context
    relevant_docs = retrieve_relevant_docs(user_prompt, k=3)
    full_input = f"Query: {user_prompt} Context: {''.join(relevant_docs)}"

    response = llm.invoke(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_input}])
    assistant_response = response.content

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").markdown(assistant_response)
