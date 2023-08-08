import sys
sys.path.append("src/langchain_agent/")

from chroma_retrieve import retrieve_chromadb
from similarity_search import similarity_search
from send_prompt import send_knowledge_infused_prompt_to_GPT
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

vectordb = retrieve_chromadb(collection_name="one_piece")

def chat(message: str) -> str:
    # Build prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": message})
    return result["result"]
