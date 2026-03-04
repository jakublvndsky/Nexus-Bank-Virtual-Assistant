# Nexus Bank Virtual Assistant

A conversational AI assistant for **Nexus Bank S.A.** (fictional Polish bank). It answers retail clients’ questions about accounts, fees, interest rates, and currency exchange using the bank’s Terms and Conditions and live NBP exchange rates.

## Features

- **RAG (Retrieval-Augmented Generation)** — Vector store over the official *General Terms and Conditions of Retail Banking Services of Nexus Bank S.A.* (PDF). Answers are grounded in the document.
- **Currency exchange** — Tool that fetches current average exchange rates from the National Bank of Poland (NBP) API (PLN → EUR, USD, GBP, CHF, etc.) and applies the 2.50 PLN commission from the Terms.
- **Compliant behaviour** — No financial advice, no handling of passwords/PINs, no access to account balances; suggests contacting support when needed.
- **Interactive chat** — Run from the terminal; streamed replies in the language used by the user.

## Tech Stack

- **Python 3**
- **LangChain** — Agent, tools, RAG
- **LangChain OpenAI** — Chat model (GPT) and embeddings
- **In-memory vector store** — Document chunks with OpenAI embeddings
- **NBP API** — Exchange rates
- **python-dotenv** — Environment variables

## Project Structure

```
Assesment/
├── main.py              # Entry point: chat loop
├── app/
│   ├── agent.py         # Agent definition, system prompt, LLM
│   ├── tools.py         # currency_exchange, retrive_from_vector_db
│   └── vector_db.py     # PDF loader, text splitter, vector store init
├── data/
│   └── Nexus Bank Terms and Conditions.pdf
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and enter the project

```bash
cd Assesment
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your OpenAI API key.

### 4. Run the assistant

From the project root:

```bash
python main.py
```

Then type your questions in the prompt. Use `exit`, `q`, or `quit` to end the session.

### 5. Run the assistant in Streamlit (web UI)

From the project root:

```bash
streamlit run streamlit_app.py
```

This will start a web application with a chat interface and visible conversation history in your browser.

## Usage Examples

- *“What are the fees for a Standard Account?”* — Uses the vector DB (Terms and Conditions).
- *“How much does it cost to withdraw cash from a foreign ATM with a Standard Account?”* — Combines fees from the document and, if relevant, current EUR rate and commission.
- *“What is the current EUR exchange rate?”* or *“How much is 100 EUR in PLN?”* — Uses the NBP tool and applies the 2.50 PLN commission.

## Notes

- The vector store is **in-memory** and is filled once at startup from `data/Nexus Bank Terms and Conditions.pdf`. Initialization happens in `main.py` (not on every message), so the PDF is loaded and indexed when you run the app.
- Dependencies are listed in `requirements.txt` (including `langchain-openai`); run `pip install -r requirements.txt` to install everything.
- Exchange rates are from the public [NBP API](http://api.nbp.pl/); no API key required.
- This is a demo for a fictional bank (Codecademy assessment). Do not use for real banking or financial decisions.
