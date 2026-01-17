import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# These specific paths are the 2026 standard for modular LangChain
# Change these lines in generator.py
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()

def get_llm():
    """Initializes the Groq LLM using the current 2026 standard model."""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        # Change this line from 3.1 to 3.3
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1
    )

def create_rag_chain(retriever):
    """Combines LLM and Retriever into a single RAG chain."""
    llm = get_llm()
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. \n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # This creates the logic to feed the PDF chunks into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # This connects your retriever (from retrieval.py) to the generator
    return create_retrieval_chain(retriever, question_answer_chain)