# Legal Compliance Hackathon

### 1. **Set Up Your Local Environment**

1. **Install Python**:
   - Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).
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
   pip install transformers torch requests huggingface-hub python-dotenv

   ```

### 2. **Set Up Hugging Face Authentication**

1. **Generate a Hugging Face API Token**:
   - Go to your Hugging Face account settings and generate an API token if you don't have one already.
2. **Authenticate with Hugging Face**:
   - Use the token to authenticate your local environment.
3. **Create .env file and type:**

```bash
HUGGINGFACE_TOKEN=your_hugging_face_api_token

CRIMINAL_KEY=hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA
CIVIL_KEY=gO(oHjM6!qyc7$iRWV@nD0GvIaTw13t
HIGH_CRIMINAL_KEY=NGaX&K2P5G39cpKHFNSrMyK=wzPJJSb
HIGH_CIVIL_KEY=P7zdKSLg2_vC9Q$)bUFGqtfX6kTcpNJ629

```

### 3. Run the Script

```bash
python llama3.py

```
