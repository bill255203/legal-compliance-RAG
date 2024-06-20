# Legal Compliance Hackathon

This project demonstrates a pipeline to generate a prompt using a Language Learning Model (LLM), fetch documents from an API, format the context and query, and get a response from the LLM using the provided context. The project uses the llama_index module to format the prompts and integrates with the CopyToaster API for document retrieval.

**What the Files Does**:

1. Document Fetching: Fetch relevant documents using the CopyToaster API.
2. Content Extraction: Extract document contents from the fetched results.
3. Template Formatting: Use the PromptTemplate to format the context and query into the desired template.
4. Final Prompt Generation: Send the formatted template to the LLM to get the final response.
5. Print Final Response: Output the final response from the LLM.
6. Running the Script
   Update the main.py file with your CopyToaster API key before running the script.

### 1. **Set Up Your Local Environment**

1. **Git Clone Project**:

   ```bash
   git clone https://github.com/bill255203/ollama.git
   cd ollama
   ```

2. **Create a Virtual Environment**:

   - It's a good practice to create a virtual environment for your project to manage dependencies.

   ```bash
   python -m venv myenv

   ```

3. **Activate the Virtual Environment**:

   - On Windows:

     ```bash
     myenv\Scripts\activate

     ```

   - On macOS/Linux:

     ```bash
     source myenv/bin/activate

     ```

4. **Install Required Packages**:

   - Use `pip` to install the required libraries.

   ```bash
   pip install transformers torch requests huggingface-hub python-dotenv llama_index ollama

   ```

5. **Install Required Models**:

   - Use `ollama` to install the model ( or if you want to add others), and patiently wait for installation.

   ```bash
   ollama run llama2-chinese

   ```

   go to `https://ollama.com/library` for more information and models

### 2. **Set Up Hugging Face Authentication**

1. **Generate a Hugging Face API Token**:
   - Go to your Hugging Face account settings and generate an API token if you don't have one already.
2. **Authenticate with Hugging Face**:
   - Use the token to authenticate your local environment.
3. **Create .env file and copy and paste the below lines:**

```bash
HUGGINGFACE_TOKEN=your_hugging_face_api_token

CRIMINAL_KEY=hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA
CIVIL_KEY=gO(oHjM6!qyc7$iRWV@nD0GvIaTw13t
HIGH_CRIMINAL_KEY=NGaX&K2P5G39cpKHFNSrMyK=wzPJJSb
HIGH_CIVIL_KEY=P7zdKSLg2_vC9Q$)bUFGqtfX6kTcpNJ629

```

### 3. Run the Script for ollama.py (main script)

```bash
python ollama.py

```

## Streamlit chatbot

The main.py contains a cool chatbot that is like a cool chatgpt, you can play with it by running:

```bash
C:\Users\USER\AppData\Roaming\Python\Python311\Scripts\streamlit run main.py
```

### Backup code for venv setup

```bash
c:\Users\USER\OneDrive\桌面\llama3\.venv\Scripts\python.exe -m pip install ollama
```

# Project Structure

ollama/
├── llama3.py # My experimental script for backup code
├── main.py # Run the interesting chatbot using the model you downloaded
├── ollama.py # Main script to run the main pipeline
├── README.md # This README file
└── requirements.txt # List of Python dependencies
