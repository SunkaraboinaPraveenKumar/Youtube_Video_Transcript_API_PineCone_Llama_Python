import os
import tempfile
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_session import Session
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from fpdf import FPDF
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Session Configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
Session(app)

# Load API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize LLAMA model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "webbot"
index = pc.Index(index_name)
spec = ServerlessSpec(cloud='aws', region='us-east-1')


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# YouTube Video Processing
@app.route('/process_video', methods=['POST'])
def process_video():
    video_url = request.form.get('video_url')
    if not video_url:
        return jsonify({"error": "Please enter a valid YouTube URL."}), 400

    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
        docs = loader.load()
        if docs:
            session['transcript'] = docs[0].page_content
            return jsonify({"transcript": session['transcript']}), 200
        else:
            return jsonify({"error": "No transcript found for the video."}), 400
    except Exception as e:
        app.logger.error(f"Failed to process video: {e}")
        return jsonify({"error": f"Failed to process video: {e}"}), 500


# Splitting transcript into chunks
def split_transcript(transcript_, chunk_size=1000):
    return [transcript_[i:i + chunk_size] for i in range(0, len(transcript_), chunk_size)]


# Storing transcript in Pinecone
@app.route('/store_transcript', methods=['POST'])
def store_transcript():
    transcript = session.get('transcript')
    if not transcript:
        return jsonify({"error": "No transcript to store."}), 400

    try:
        transcript_chunks = split_transcript(transcript)
        for i, chunk in enumerate(transcript_chunks):
            vectors = [{
                "id": f"transcript_chunk_{i}",
                "values": HuggingFaceEmbeddings().embed_documents([chunk])[0],
                "metadata": {"chunk": chunk}
            }]
            index.upsert(vectors)

        return jsonify({"message": "Transcript chunks stored successfully!"}), 200
    except Exception as e:
        app.logger.error(f"Failed to store transcript: {e}")
        return jsonify({"error": f"Failed to store transcript: {e}"}), 500


# Retrieving relevant content from Pinecone
def retrieve_relevant_docs(query_input, k=3):
    try:
        query_embedding = HuggingFaceEmbeddings().embed_query(query_input)
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_values=False,
            include_metadata=True
        )
        return [result['metadata']['chunk'] for result in results['matches']] if results['matches'] else ["No relevant context found."]
    except Exception as e:
        app.logger.error(f"Failed to retrieve relevant docs: {e}")
        return ["Error retrieving relevant context."]


# Summarizing Transcript
@app.route('/summarize_transcript', methods=['POST'])
def summarize_transcript():
    transcript = session.get('transcript')
    if not transcript:
        return jsonify({"error": "No transcript available for summarization."}), 400

    try:
        summarize_prompt = PromptTemplate.from_template("""
        You are a helpful assistant that summarizes transcripts:
        {context}
        Summarize the context in simple and clear language with examples.
        """)

        if len(transcript) > 10000:
            transcript_chunks = split_transcript(transcript, 10000)
            summaries = []
            for chunk in transcript_chunks:
                summary_chain = summarize_prompt | llm | RunnableLambda(lambda message: message.content.strip())
                summaries.append(summary_chain.invoke({"context": chunk}))
            session['summary'] = " ".join(summaries)
        else:
            summary_chain = summarize_prompt | llm | RunnableLambda(lambda message: message.content.strip())
            session['summary'] = summary_chain.invoke({"context": transcript})

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Video Transcript:\n{transcript}\n\nSummary:\n{session['summary']}")
        pdf_path = os.path.join(tempfile.gettempdir(), "video_summary.pdf")
        pdf.output(pdf_path)

        return jsonify({"summary": session['summary'], "pdf_path": pdf_path}), 200
    except Exception as e:
        app.logger.error(f"Failed to summarize transcript: {e}")
        return jsonify({"error": f"Failed to summarize transcript: {e}"}), 500


# Download PDF
@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_path = os.path.join(tempfile.gettempdir(), "video_summary.pdf")
    try:
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Failed to download PDF: {e}")
        return jsonify({"error": f"Failed to download PDF: {e}"}), 500


# Chat with LLAMA
@app.route('/chat', methods=['POST'])
def chat():
    user_prompt = request.json.get('user_prompt')
    if not user_prompt:
        return jsonify({"error": "No user prompt provided."}), 400

    transcript = session.get('transcript', "")
    context = f"Relevant Transcript Context:\n{transcript[:500]}" if transcript else ""
    relevant_docs = retrieve_relevant_docs(user_prompt, k=3)
    full_input = f"Query: {user_prompt} Context: {''.join(relevant_docs)}"

    try:
        response = llm.invoke(
            [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_input}]
        )
        assistant_response = response.content
        return jsonify({"assistant_response": assistant_response}), 200
    except Exception as e:
        app.logger.error(f"Failed to generate chat response: {e}")
        return jsonify({"error": f"Failed to generate chat response: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
