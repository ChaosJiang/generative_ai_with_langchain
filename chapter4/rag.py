from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages

from chapter4.llms import chat_model
from chapter4.retriever import DocumentRetriever

system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some corporate document snippets, write documentation."
    "If none of the documents is relevant to the question, "
    "mention that there's no relevant document, and then "
    "answer the question to the best of your knowledge."
    "\n\nHere are the corporate documents: "
    "{context}"
)

retriever = DocumentRetriever()
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str
    issues_report: str
    issues_detected: bool
    messages: Annotated[list, add_messages]


def retrieve(state: State) -> dict[str, Document]:
    """Retrieve relevant documents based on the question."""
    retrieved_docs = retriever.invoke(state["messages"][-1].content)
    print(retrieved_docs)
    return {"context": retrieved_docs}


def generate(state: State):
    if not state["context"]:
        # No documents were found or processed
        response = chat_model.invoke(
            [
                {
                    "role": "system",
                    "content": "You're a helpful AI assistant. The user has asked a question, but no relevant documents were found in the system.",
                },
                {
                    "role": "user",
                    "content": f"Question: {state['messages'][-1].content}\n\nPlease respond to this question based on your general knowledge, and politely mention that no relevant corporate documents were found.",
                },
            ]
        )
    else:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["messages"][-1].content, "context": docs_content}
        )
        response = chat_model.invoke(messages)

    print(response.content)
    return {"answer": response.content}


def double_check(state: State):
    result = chat_model.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"Review the following project documentation for compliance with our corporation standards."
                    f"Return 'ISSUES FOUND' followed by any issues detected or 'NO ISSUE': {state['answer']}"
                ),
            }
        ]
    )
    content = result.content
    if "</think>" in content:
        actual_response = content.split("</think>", 1)[1].strip()
    else:
        actual_response = content.strip()

    if "ISSUES FOUND" in actual_response.upper():
        print("issues_detected")
        return {
            "issues_report": actual_response.split("ISSUES FOUND", 1)[1].strip(),
            "issues_detected": True,
        }
    print("no issue detected")
    return {"issues_report": "", "issues_detected": False}


def doc_finalizer(state: State):
    """Finalize documentation by integration feedback."""
    if "issue_detected" in state and state["issues_detected"]:
        response = chat_model.invoke(
            [
                {
                    "role": "user",
                    "content": (
                        f"Revise the following documentation to address these feedback points: {state['issues_report']}\n"
                        f"Original Documentation: {state['answer']}\n"
                        f"Always return the full revised document, even if no changes are needed."
                    ),
                }
            ]
        )
        return {"messages": [AIMessage(response.content)]}
    return {"messages": [AIMessage(state["answer"])]}


graph_builder = StateGraph(State).add_sequence(
    [retrieve, generate, double_check, doc_finalizer]
)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("doc_finalizer", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# # Use the thread_id directly in the config
# input_message = [HumanMessage("What's the square of 10?")]
# response = graph.invoke({"messages": input_message})
# print(response["messages"][-1].content)
