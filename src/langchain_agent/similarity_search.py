from langchain.vectorstores import Chroma

def similarity_search(
    query: str,
    vectordb: Chroma,
    top_k: int = 5,
    ):
    """
    Retrieve the top_k most similar messages to the given message from the given collection.

    Args:
        message (str): Message to be compared to the messages in the collection.
        vectordb (Chroma): Chroma object containing the collection.
        top_k (int, optional): Number of similar messages to be returned. Defaults to 5.
    
    Returns:
        List[str]: List of the top_k most similar messages to the given message from the given collection.
    """
    similar_messages = vectordb.similarity_search(
        query=query,
        top_k=top_k,
        )
    return similar_messages