import sys
sys.path.append("src/langchain_agent/")

from chroma_retrieve import retrieve_chromadb
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

vectordb = retrieve_chromadb(collection_name="one_piece")

def chat(message: str, chat_history: list) -> str:
    # Build prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. Please be as specific as possible, and don't include information that is not corroborated by the context or your general knowledge.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.25}),# Retrieve more documents with higher diversity- useful if your dataset has many similar documents
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result_ConversationalRetrievalChain = qa_ConversationalRetrievalChain({"question": message, "chat_history": chat_history})
    print(result_ConversationalRetrievalChain)
    chat_history.append((message, result_ConversationalRetrievalChain["answer"]))

    return result_ConversationalRetrievalChain["answer"], chat_history
