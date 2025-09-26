import streamlit as st
from langchain_core.messages import HumanMessage

from chapter4.document_loader import DocumentLoader
from chapter4.rag import config, graph, retriever

st.set_page_config(page_title="Corporate Documentation Manager", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

for message in st.session_state.chat_history:
    print(f"message: {message}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process all uploaded files
if st.session_state.uploaded_files:
    retriever.add_uploaded_docs(st.session_state.uploaded_files)


def process_message(message):
    """Assistant response."""
    response = graph.invoke({"messages": HumanMessage(message)}, config=config)
    return response["messages"][-1].content


st.markdown(
    """
# 📄 CorpDocs with Citations

CorpDocs is your corporate documentation assistant. This tool generates detailed project documentation,
verifies compliance with corporate standards, and integrates human feedback when necessary. Finally,
it retrieves and attaches source citations to the final document.

**Workflow:**
1. **Generate Documentation:** Create an initial draft.
2. **Compliance Check:** Automatically review for adherence to corporate guidelines.
3. **Human Feedback:** If issues are detected, provide corrective feedback.
4. **Finalize Document:** Produce the revised document.
5. **Add Citations:** Append source citations to the document.

If you like this application, please give us a 5-star review on [Amazon](https://amzn.to/3X1xQbn)!
"""
)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Chat Interface")
    # React to user input
    if user_message := st.chat_input("Enter your message:"):
        with st.chat_message("User"):
            st.markdown(user_message)

        st.session_state.chat_history.append({"role": "User", "content": user_message})
        response = process_message(user_message)
        with st.chat_message("Assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "Assistant", "content": response})

with col2:
    st.subheader("Document Management")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=list(DocumentLoader.supported_extensions),
        accept_multiple_files=True,
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file)
