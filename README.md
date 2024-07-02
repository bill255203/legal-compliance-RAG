# Legal Compliance Hackathon

This project demonstrates a pipeline to generate a prompt using a Language Learning Model (LLM), fetch documents from an API, format the context and query, and get a response from the LLM using the provided context. The project integrates various tools for document retrieval, vectorization, and querying.

# Setup Instructions

## 1. Set Up Your Local Environment

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

For more information and models, visit [Ollama Library](https://ollama.com/library).

### 1.5. Create `.env` File

Create a `.env` file in the project root and add the following lines:

```bash
HUGGINGFACE_TOKEN=your_hugging_face_api_token

CRIMINAL_KEY=hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA
CIVIL_KEY=gO(oHjM6!qyc7$iRWV@nD0GvIaTw13t
HIGH_CRIMINAL_KEY=NGaX&K2P5G39cpKHFNSrMyK=wzPJJSb
HIGH_CIVIL_KEY=P7zdKSLg2_vC9Q$)bUFGqtfX6kTcpNJ629

```

## 2. Run the Scripts

### 2.1. Run the New Pinecone Script

```bash
python create-pinecone.py
python success.py
```

### 3. Run the Streamlit Chatbot (`main.py`)

The `main.py` contains a chatbot. Run it using Streamlit:

```bash
C:\\Users\\USER\\AppData\\Roaming\\Python\\Python311\\Scripts\\streamlit run main.py

```
