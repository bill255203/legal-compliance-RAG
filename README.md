# Legal Compliance Hackathon

This project demonstrates a pipeline to generate a prompt using a Language Learning Model (LLM), fetch documents from an API, format the context and query, and get a response from the LLM using the provided context. The project integrates various tools for document retrieval, vectorization, and querying. It is done by creating a Pinecone database from data in a zip file, extracting content from the documents, translating it to English, feeding it to the Groq Llama model, and translating the output back to Mandarin.

## Setup Instructions

### PreSetup: Retrieve and Prepare Laws.zip File

1. Download the laws.zip file from the official provided documents.
2. Unzip the laws.zip file into the C:// directory.
3. Warning: Leave only the first 5 to 10 folders in the unzipped laws folder and delete the rest to avoid incurring cost.

### 1. Set Up a Google Cloud Project

### 1.1. Create a Google Cloud Project:

1. Go to the Google Cloud Console [here](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Make sure your Google Cloud project has billing enabled. You will need to provide your credit card information to use the Cloud Translation API.
4. Enable the Cloud Translation API for your project.

### 1.2. Create a Service Account and Download Credentials:

1. In the Google Cloud Console, go to "IAM & Admin" > "Service Accounts".
2. Click "Create Service Account".
3. Give it a name and grant it the "Cloud Translation API User" role.
4. Create a key for this service account:
   - Click on the newly created service account.
   - Go to the "Keys" tab.
   - Click "Add Key" > "Create New Key".
   - Choose JSON as the key type.
   - Click "Create" and download the JSON file.

### 1.3. Add Pinecone API Key

1. Sign up for a Pinecone account `https://www.pinecone.io/`.
2. Create an API key in the Pinecone dashboard.
3. Add the API key to your `.env` file as `PINECONE_API_KEY`.

### 1.4. Add Groq API Key

1. Sign up for a Groq account.
2. Create an API key in the Groq dashboard.
3. Add the API key to your `.env` file as `GROQ_API_KEY`.

### 2. Set Up Your Local Environment

### 2.1. Git Clone Project

```bash
git clone <https://github.com/bill255203/legal-compliance-RAG.git>
cd legal-compliance-RAG

```

### 2.2. Create a Virtual Environment

It's a good practice to create a virtual environment for your project to manage dependencies.

```bash
python -m venv .venv

```

### 2.3. Activate the Virtual Environment

On Windows:

```bash
.venv/Scripts/activate

```

On macOS/Linux:

```bash
source .venv/bin/activate

```

### 2.4. Install Required Packages

Use pip to install the required libraries.

```bash
pip install -r requirements.txt

```

### 3. Configure Environment Variables

### 3.1. Create a .env File

Create a `.env` file in the project root and add the following lines:

```less
CRIMINAL_KEY=hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA
CIVIL_KEY=gO(oHjM6!qyc7$iRWV@nD0GvIaTw13t
HIGH_CRIMINAL_KEY=NGaX&K2P5G39cpKHFNSrMyK=wzPJJSb
HIGH_CIVIL_KEY=P7zdKSLg2_vC9Q$)bUFGqtfX6kTcpNJ629

PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key

GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

```

### 4. Run the Scripts

### 4.1. Run the Pinecone Script

```bash
python create-pinecone.py
python success.py

```
