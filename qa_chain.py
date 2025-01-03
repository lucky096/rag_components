import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory

from config import Config


store = dict()

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question.
If the answer is not found within the context, state the answer can not be found.
Prioritize concise responses (maximum of 3 sentences) and use a list where applicable.
The contextual information is organized with the most recent source at the top.
Each source is separated  by a separator (---).

Context:
{context}

Use markdown  formatting wherever appropriate.
"""

def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)

def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")
    
    return remove_links("\n".join(texts))

def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    prompt = ChatPromptTemplate.from_message(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (RunnablePassthrough.assign(
        context=itemgetter("question") | retriever.with_config({"run_name": "context_retriever"}) | format_documents
    ) | prompt | llm)

    return RunnableWithMessageHistory(
        chain,
        get_session_history, 
        input_messages_key="question", 
        history_messages_key="chat_history"
        ).with_config({"run_name": "qa_chain"})

async def ask_question(chain: Runnable, question: str, session_id: str):
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "qa_chain"], #output of context retriever and qa_chain
    ):
        event_type = event["event"]
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        elif event_type == "on_chain_stream":
            yield event["data"]["chunk"].content #streaming the chunk of data