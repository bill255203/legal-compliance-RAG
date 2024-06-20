import requests
import json
from llama_index.core import PromptTemplate

# Step 1: Generate a prompt using the API
def generate_prompt(model, prompt, stream=False):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data['response']
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None

# Step 2: Fetch documents from the CopyToaster API
def fetch_documents_from_copytoaster(input_str, count):
    url = "https://api.droidtown.co/CopyToaster/API/"
    payload = {
        "username": "legaltech@droidtown.co",  # Do not change this field
        "copytoaster_key": "hZYNiEDyd5VCpP(FJS1w@UeRsgLr$QA",  # Replace with your desired court key
        "category": "臺灣台北地方法院",  # Input the court name you want to search
        "input_str": input_str,  # Input the sentence you want to search
        "count": count  # Optional, default is 15
    }
    response = requests.post(url, json=payload).json()
    if response["status"]:
        print("Debug: Inside if condition")
        # print("Response Results:", response["results"])
        return response["results"]
    else:
        return []

# Step 3: Extract document contents from fetched results
def extract_document_content(documents):
    contents = []
    for doc in documents:
        contents.append(doc['document'])
    return ' '.join(contents)

# Step 4: Combine the content to form a complete template string and ask it to the generate prompt function
def main():
    # Initialize the parameters
    model = "llama2-chinese"
    initial_prompt = "用LINE恐嚇取財"

    # Fetch documents using the generated prompt
    documents = fetch_documents_from_copytoaster(initial_prompt, 3)
    print(f"RAG docs: {documents}")
    
    document_content = extract_document_content(documents)
    print(f"Extracted RAG content: {document_content}")
    
    # Define the template
    template = (
        "We have provided example legal documents below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please generate legal compliance document in chinese which the topic is about {query_str}\n"
    )
    qa_template = PromptTemplate(template)

    # Format the context and query
    context_str = document_content
    formatted_prompt = qa_template.format(context_str=context_str, query_str=initial_prompt)

    # Generate the final prompt using the formatted template
    final_response = generate_prompt(model, formatted_prompt)
    print("Final Response:\n", final_response)

if __name__ == "__main__":
    main()
