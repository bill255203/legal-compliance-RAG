# Legal Compliance Hackathon

This project demonstrates a pipeline to generate a prompt using a Language Learning Model (LLM), fetch documents from an API, format the context and query, and get a response from the LLM using the provided context. The project integrates various tools for document retrieval, vectorization, and querying.

## Project Structure

```
legal-compliance-RAG/
├── llama3.py               # Experimental script for backup code
├── main.py                 # Run the interesting chatbot using the model downloaded in Ollama
├── llama2-chinese.py       # Main script to run and translate to Chinese
├── vectorize-docs.py       # Latest version of vectorization file
├── README.md               # This README file
└── requirements.txt        # List of Python dependencies

```

## Setup Instructions

### 1. Set Up Your Local Environment

### 1.1. Git Clone Project

```bash
git clone <https://github.com/bill255203/legal-compliance-RAG.git>
cd legal-compliance-RAG

```

### 1.2. Create a Virtual Environment

It's a good practice to create a virtual environment for your project to manage dependencies.

```bash
python -m venv .venv

```

### 1.3. Activate the Virtual Environment

- On Windows:

  ```bash
  .venv/Scripts/activate

  ```

- On macOS/Linux:

  ```bash
  source .venv/bin/activate

  ```

### 1.4. Install Required Packages

Use `pip` to install the required libraries.

```bash
pip install -r requirements.txt

```

### 2. Install Required Models

Use `ollama` to install the model (or other models if needed) and patiently wait for installation.

```bash
ollama run llama2-chinese

```

For more information and models, visit [Ollama Library](https://ollama.com/library).

### 3. Set Up Hugging Face Authentication

### 3.1. Generate a Hugging Face API Token

Go to your Hugging Face account settings and generate an API token if you don't have one already.

### 3.2. Authenticate with Hugging Face

Use the token to authenticate your local environment.

### 3.3. Create `.env` File

Create a `.env` file in the project root and add the following lines:

```bash
HUGGINGFACE_TOKEN=your_hugging_face_api_token

CRIMINAL_KEY=hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA
CIVIL_KEY=gO(oHjM6!qyc7$iRWV@nD0GvIaTw13t
HIGH_CRIMINAL_KEY=NGaX&K2P5G39cpKHFNSrMyK=wzPJJSb
HIGH_CIVIL_KEY=P7zdKSLg2_vC9Q$)bUFGqtfX6kTcpNJ629

```

### 4. Run the Scripts

### 4.1. Run the Main Script (`ollama.py`)

```bash
python llama2-chinese.py

```

### 4.2. Run the Streamlit Chatbot (`main.py`)

The `main.py` contains a chatbot. Run it using Streamlit:

```bash
C:\\Users\\USER\\AppData\\Roaming\\Python\\Python311\\Scripts\\streamlit run main.py

```

### 4.3. Run the New Vectorization Script (`create-faiss.py`)

- **The `create-faiss.py` script handles document vectorization and should be run to set up the vector database.**

```bash
python create-faiss.py
```

### 4.4. Run the New Pinecone Script

```bash
python create-pinecone.py
python pinecone-query.py
```

### Additional Notes

- **Ensure you update the `ollama.py` file with your CopyToaster API key before running the script.**
