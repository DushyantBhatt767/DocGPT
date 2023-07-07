# DocGPT using HuggingFace and Azure search

This is a Python code for to create chat bot on your custom knowledge-base. It allows you to upload PDF documents, extract the text from them, index the text using Azure Search, and perform document search based on user queries. Additionally, it provides a chat interface where you can ask questions and receive answers based on the indexed documents.

## Prerequisites

- Python 3.x

## Setup

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (optional but recommended) and activate it.

3. Install the dependencies as mentioned in the prerequisites section.

4. Create a `.env` file in the project directory and set the following environment variables:

   - `AZURE_SEARCH_ENDPOINT`: The endpoint URL of your Azure Search service.
   - `AZURE_SEARCH_KEY`: The API key for your Azure Search service.
   - `AZURE_SEARCH_INDEX`: The name of the index in Azure Search where the documents will be indexed.
   - `HUGGINGFACEHUB_API_TOKEN`: Huggingface API token to use huggingface models

## Running the Application

To run the application, execute the following command in the project directory:
```
streamlit run app.py
```
Once the application is running, it will open in your default web browser, and you can interact with it.

## Usage

1. The web application will open with a header "Ask anything to your documents" and a text input field labeled "Ask a question about your documents".

2. Upload your PDF documents using the file uploader provided in the sidebar. You can upload multiple files.

3. Click on the "Process" button to extract the text from the uploaded PDFs and index them in Azure Search. The processing may take some time depending on the size and number of documents.

4. Once the processing is complete, you can enter your question in the "Ask a question about your documents" text input field and press Enter or click outside the input field.

5. The application will display the conversation between you and the chatbot. The chatbot will provide answers based on the indexed documents.

6. You can continue asking questions and receiving answers in the chat interface.

