import pandas as pd
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document


# ToDo:
# 1- Reads and preprocesses customer service conversations from a CSV file.
dataframe = pd.read_csv('customer_service_conversations.csv')

# 2- Generates embeddings for customer messages using sentence-transformers
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# 3- Stores embeddings in a vector database (FAISS) for fast similarity search
raw_documents = [Document(page_content=text) for text in dataframe["agent_response"].tolist()]

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
vector_store = FAISS.from_documents(documents, embeddings)

# Model
model = ChatOllama(model="llama3.2", temperature=0.3)

# 4- Implements a function to retrieve the top 3 most relevant responses based on a user query.
def retrieve_responses(query: str) -> Any:
    # Search for the most relevant documents in the vector database
    docs = vector_store.similarity_search(query, k=3)

    # Formats the context from the found documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Creates a suitable message for the model
    prompt = f"""
    Based on the following customer service conversations:

    {context}

    Answer the question: {query}
    """

    # Generates a response using the model
    response = model.invoke([HumanMessage(content=prompt)])

    return response

def retrieve_top_3_documents(query: str) -> Any:

    docs = vector_store.similarity_search(query, k=3)

    for n in range(len(docs)):
        print(docs[n].page_content)

    return docs

# Example query
query = "How can I contact the support team?"
response = retrieve_top_3_documents(query)



