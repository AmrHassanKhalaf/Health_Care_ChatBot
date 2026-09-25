# Health Care ChatBot

An educational, retrieval-augmented chatbot for **pre-marital health awareness**. The application retrieves relevant passages from a curated medical PDF collection in Pinecone, then uses Google Gemini to produce short, respectful answers about premarital testing, genetic counseling, reproductive-health awareness, and preparing for marriage.

> **Medical disclaimer:** This project is for general education only. It does not diagnose conditions, prescribe treatment, or replace advice from qualified healthcare professionals.

## Project background

Developed during the **National Telecommunication Institute (NTI) Summer Internship** in Egypt (August–September 2025). The project applies concepts from a 90-hour Natural Language Processing training program, including text preprocessing, feature engineering, Bag of Words, N-grams, TF-IDF, and NLP model workflows.

## Highlights

- Uses a Retrieval-Augmented Generation (RAG) workflow to ground responses in uploaded PDF material.
- Loads and splits PDF documents into overlapping chunks for semantic retrieval.
- Generates 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores and searches vectors in Pinecone using cosine similarity.
- Uses `gemini-2.5-flash` through LangChain for answer generation.
- Applies a safety-focused prompt: concise, neutral, privacy-aware, culturally sensitive, and educational responses.
- Includes both a Flask web chat interface and a Gradio interface.

## Architecture

```text
PDF knowledge base
       |
       v
PyPDFLoader -> text cleanup -> chunking -> Hugging Face embeddings
       |                                         |
       +--------------> Pinecone vector index <--+
                                      |
User question -> similarity retrieval (top 3) -> Gemini -> answer
```

## Repository structure

```text
Health_Care_ChatBot/
|-- app.py              # Flask application and RAG chain
|-- gradio_app.py       # Gradio chat interface and conversation state
|-- Store_index.py      # PDF ingestion and Pinecone indexing script
|-- data/
|   `-- Medical_book.pdf # Source knowledge document
|-- src/
|   |-- helper.py        # Loading, cleaning, chunking, and embeddings helpers
|   `-- prompt.py        # Medical education and safety prompt
|-- templates/chat.html  # Flask chat page
|-- static/style.css     # Flask interface styling
|-- requirements.txt     # Python dependencies
`-- test_import.py       # Basic environment/import connectivity check
```

## Prerequisites

- Python 3.10 or later
- A [Pinecone](https://www.pinecone.io/) account and API key
- A Google AI/Gemini API key
- Internet access on the first run to download the embedding model

## Installation

1. Clone the repository and enter it.

   ```bash
   git clone <your-repository-url>
   cd Health_Care_ChatBot
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   pip install langchain-huggingface pinecone
   ```

4. Create a `.env` file in the project root.

   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   GOOGLE_API_KEY=your_google_ai_api_key
   ```

   Keep this file private; it is already excluded from Git.

## Build the knowledge index

Place your medical PDF files in `data/`, then run:

```bash
python Store_index.py
```

The script reads PDFs from `data/`, removes unnecessary metadata, splits text into 500-character chunks with a 20-character overlap, creates the `medical-chatbot` index if needed, and uploads the embedded chunks to Pinecone.

## Run the application

### Gradio interface

```bash
python gradio_app.py
```

Open the local URL printed in the terminal.

### Flask interface

```bash
python app.py
```

Then open `http://localhost:10000` in your browser. Set the `PORT` environment variable to use another port.

### Important index configuration

`Store_index.py` creates the Pinecone index named `medical-chatbot`. Before running either interface, make sure the `index_name` in its application file points to the same populated index. This is currently configured separately in `app.py` and `gradio_app.py` so it can be adapted for different knowledge bases.

## Response policy

The system prompt instructs the assistant to:

- Prioritize retrieved context and avoid fabricating medical facts.
- Use a calm, respectful, non-judgmental tone.
- Give short answers with a title and key points.
- Avoid diagnoses, treatment plans, and individualized medical advice.
- Encourage professional consultation when a medical decision is involved.
- State when a question falls outside the chatbot's scope.

## Troubleshooting

| Issue | Suggested check |
| --- | --- |
| Pinecone connection fails | Confirm `PINECONE_API_KEY`, the index name, and that the index exists in the configured Pinecone region. |
| Gemini requests fail | Confirm `GOOGLE_API_KEY` and that it has access to the configured model. |
| Retrieval gives irrelevant answers | Verify that the target index was populated from the intended PDFs and that the interface uses its exact name. |
| First run is slow | The embedding model is downloaded and initialized locally on first use. |

You can also run the basic setup check:

```bash
python test_import.py
```

## Technology stack

Python, LangChain, Google Gemini, Pinecone, Sentence Transformers, Hugging Face embeddings, Flask, Gradio, PyPDF, and python-dotenv.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
