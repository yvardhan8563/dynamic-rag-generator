import streamlit as st

from frontend.api_client import APIError, call

st.set_page_config(page_title="Dynamic RAG Generator", page_icon="📚", layout="wide")
st.title("Dynamic RAG Generator")
st.caption("Upload documents, ask questions, and inspect the evidence behind each answer.")

if notice := st.session_state.pop("notice", None):
    st.success(notice)

with st.sidebar:
    st.header("Collections")
    with st.form("create_collection", clear_on_submit=True):
        name = st.text_input("New collection name", max_chars=120)
        create = st.form_submit_button("Create collection")
    if create:
        if not name.strip():
            st.error("Enter a collection name.")
        else:
            try:
                created = call("POST", "/collections", json={"name": name.strip()})
                st.session_state["selected_collection"] = created["id"]
                st.session_state["notice"] = "Collection created."
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

try:
    collections = call("GET", "/collections")
except APIError as exc:
    st.error(str(exc))
    st.stop()

if not collections:
    st.info("Create a collection in the sidebar to get started.")
    st.stop()

names = {c["id"]: c["name"] for c in collections}
if st.session_state.get("selected_collection") not in names:
    st.session_state["selected_collection"] = collections[0]["id"]
cid = st.sidebar.selectbox("Active collection", options=list(names),
                           format_func=lambda key: f"{names[key]} · {key[:8]}", key="selected_collection")
selected = next(c for c in collections if c["id"] == cid)
st.subheader(names[cid])
st.caption(f"{selected['document_count']} documents · {selected['chunk_count']} chunks")

with st.sidebar.expander("Delete collection"):
    confirmed = st.checkbox("Delete all documents in this collection", key=f"confirm-{cid}")
    if st.button("Delete collection", disabled=not confirmed):
        try:
            call("DELETE", f"/collections/{cid}")
            st.session_state.pop("last_answer", None)
            st.session_state["notice"] = "Collection deleted."
            st.rerun()
        except APIError as exc:
            st.error(str(exc))

with st.expander("Upload documents", expanded=selected["document_count"] == 0):
    files = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True,
                             key=f"uploads-{cid}")
    st.caption("PDFs need selectable text. Each file is indexed separately; the default limit is 10 MB.")
    if st.button("Index uploaded documents", disabled=not files):
        for file in files:
            with st.spinner(f"Indexing {file.name}…"):
                try:
                    result = call("POST", f"/collections/{cid}/documents",
                                  files={"file": (file.name, file.getvalue(), file.type)})
                    st.success(f"{file.name}: indexed {result['chunk_count']} chunks.")
                    st.session_state.pop("last_answer", None)
                except APIError as exc:
                    st.error(f"{file.name}: {exc}")
        st.info("Uploads processed. Refresh collection details below to update the counts.")

if st.button("Refresh collection details"):
    st.rerun()

with st.expander("Indexed documents"):
    try:
        docs = call("GET", f"/collections/{cid}/documents")
        if not docs:
            st.info("No documents indexed yet.")
        for doc in docs:
            left, right = st.columns([4, 1])
            left.write(doc["filename"])
            if right.button("Remove", key=doc["id"]):
                call("DELETE", f"/collections/{cid}/documents/{doc['id']}")
                st.session_state.pop("last_answer", None)
                st.session_state["notice"] = "Document removed."
                st.rerun()
    except APIError as exc:
        st.error(str(exc))

with st.form("question"):
    question = st.text_area("Ask a question about this collection", max_chars=4000)
    top_k = st.slider("Maximum evidence passages", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")
if submitted:
    st.session_state.pop("last_answer", None)
    if not question.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Finding evidence and preparing an answer…"):
            try:
                answer = call("POST", f"/collections/{cid}/query", json={"question": question, "top_k": top_k})
                st.session_state["last_answer"] = (cid, answer)
            except APIError as exc:
                st.error(str(exc))

saved = st.session_state.get("last_answer")
if saved and saved[0] == cid:
    answer = saved[1]
    st.subheader("Answer")
    # Render untrusted LLM text without Markdown links/images or executable HTML.
    if answer["status"] == "insufficient_evidence":
        st.info(answer["answer"])
    else:
        st.text(answer["answer"])
    for citation in answer["citations"]:
        location = ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in citation["location"].items())
        with st.expander(f"[{citation['source_id']}] {citation['filename']} — {location}"):
            st.text(citation["excerpt"])
            st.caption(f"Similarity: {citation['score']:.3f} · Chunk: {citation['chunk_id']}")
