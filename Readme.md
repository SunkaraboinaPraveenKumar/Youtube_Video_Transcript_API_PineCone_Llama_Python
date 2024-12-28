# 🦙 LLAMA 3.1 Video & Chat Assistant with Pinecone

Welcome to **LLAMA 3.1 Video & Chat Assistant**, a powerful tool that integrates video transcript analysis, summarization, and real-time chat with the assistant using Pinecone for vector database storage. This project allows you to extract insights from YouTube videos, summarize them efficiently, and chat with the assistant using relevant contextual data stored in Pinecone.

## 🚀 Features

- 🎥 **YouTube Video Analysis**: Automatically fetch and analyze YouTube video transcripts.
- 📝 **Summarization**: Summarize long video transcripts into concise summaries using LLAMA 3.1.
- 💬 **Real-time Chat**: Engage in conversations with the LLAMA assistant, with Pinecone retrieving relevant context from video transcripts.
- 🧠 **Pinecone Integration**: Store and retrieve vectorized transcript data for enhanced contextual conversation.
- 📄 **Export Summaries**: Download the generated summaries as a PDF for easy reference.

## 🛠️ Tech Stack

- **LLAMA 3.1**: A large language model for summarization and chat.
- **Pinecone**: A vector database used for storing and retrieving transcript embeddings.
- **Streamlit**: A framework for building interactive web apps.
- **FPDF**: Used for generating downloadable PDFs of the transcript and summary.
- **LangChain**: For integrating and running LLMs in a streamlined manner.
- **HuggingFaceEmbeddings**: To embed transcript data into vectors for Pinecone.

## 📦 Setup & Installation

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8 or higher
- A YouTube API key (for fetching transcripts)
- Pinecone API key
- GROQ API key (for accessing the LLAMA model)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llama-3.1-video-chat-assistant.git

cd llama-3.1-video-chat-assistant


2. Install Dependencies

pip install -r requirements.txt

3. Set Up Environment Variables
Create a .env file in the root directory and add the following:


GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key


4. Run the Application

streamlit run app.py

Now, the application will be available on http://localhost:8501 where you can analyze YouTube videos, generate summaries, and chat with the assistant!

🖥️ How to Use
1. Process a YouTube Video
Enter the YouTube video URL in the input field and click Process Video.
The transcript will be fetched and displayed.
2. Store Transcript in Pinecone
After fetching the video transcript, click Store Transcript in Vector DB to store the transcript chunks in Pinecone.
The transcript will be embedded and stored in Pinecone for later retrieval during chat interactions.
3. Summarize the Transcript
Click Summarize Transcript to generate a concise summary of the video.
The summary will be displayed on the screen and can be downloaded as a PDF.
4. Chat with LLAMA
Enter a prompt in the chat box to ask LLAMA about the video or anything else.
The assistant will retrieve relevant context from Pinecone to provide more insightful answers.
📄 Example Output
YouTube Video Transcript
css
Copy code
The transcript of the video will be displayed here after fetching.
Summary
css
Copy code
A concise summary of the video will be generated here after summarization.
Chat Interaction
plaintext
Copy code
User: What is the main point of the video?

LLAMA: The main point of the video is to explain how neural networks work by demonstrating their applications in real-life scenarios...
📚 Libraries Used
LangChain: To handle the integration of LLMs and the summarization logic.
Pinecone: For storing and retrieving vector embeddings of video transcripts.
Streamlit: For creating an interactive web interface.
FPDF: For generating PDF files containing transcripts and summaries.
Hugging Face: For embedding transcripts into vectors using HuggingFaceEmbeddings.
💡 Future Improvements
Multi-Video Analysis: Support for multiple video analyses and comparisons.
Real-time Summarization: Enable real-time streaming and summarization of video content.
Customizable Chat Behavior: Allow users to fine-tune the chat experience by selecting different models or modes (e.g., conversational, factual).
🤝 Contributing
Contributions are welcome! If you'd like to contribute to this project, feel free to submit a pull request or open an issue on GitHub.

Fork the repository.
Create your feature branch: git checkout -b feature/my-feature.
Commit your changes: git commit -m 'Add my feature'.
Push to the branch: git push origin feature/my-feature.
Open a pull request.
🛡️ License
This project is licensed under the MIT License. See the LICENSE file for details.

🌟 Acknowledgements
Special thanks to the developers of LLAMA, Pinecone, and LangChain for their awesome tools!
Thanks to the open-source community for their continuous support.
Feel free to explore, modify, and enhance this project. Happy coding! 😊

