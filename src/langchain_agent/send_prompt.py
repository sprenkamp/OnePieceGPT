import os
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


def send_knowledge_infused_prompt_to_GPT(message: str, top_k_similar_list: list, model_version: str = "gpt-3.5-turbo") -> str:
    llm = ChatOpenAI(
            temperature=0,  # Make as deterministic as possible
            model_name=model_version,
        )
    top_k_similar_str = "\n".join(top_k_similar_list)
    messages = [
    SystemMessage(
        content=f"""You will obtain a message containing a request from a refugee. We already filtered the messages in our database and found the top {len(top_k_similar_list)} most similar messages to the given message. Please read the messages and use the information to write a response to the refugee.
        Please solely return the answer in the language that the refugee uses to communicate with you. If the refugee speaks Ukrainian you answer in Ukrainian, if the refugee speaks English you answer in English, etc.
        """
    ),
    HumanMessage(
        content=f"""Here is the message from the refugee: {message}
        Here are the top {len(top_k_similar_list)} most similar messages from our database: {top_k_similar_str}
        """
    ),
    ]
    output = llm(messages)
    return output.content