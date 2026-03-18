import os
import base64
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import backend


# -----------------------------
# Helpers
# -----------------------------

def display_image_base64(img):
    image_data = base64.b64decode(img)
    st.image(image_data, use_container_width=True)


def display_table(table):
    if hasattr(table.metadata, "text_as_html"):
        st.markdown(table.metadata.text_as_html, unsafe_allow_html=True)


def show_pdf_page(file, page_number):
    reader = PdfReader(file)
    page = reader.pages[page_number - 1]
    text = page.extract_text()

    st.subheader(f"📄 Page {page_number}")
    st.text(text[:2000])


# -----------------------------
# Main App
# -----------------------------

def main():

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    st.set_page_config(
        page_title="Multimodal RAG Research Assistant",
        layout="wide"
    )

    st.title("📄 Multimodal RAG Research Assistant")
    st.write("Upload a research paper and ask questions.")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    # --------------------------------
    # Process PDF
    # --------------------------------

    if pdf and "rag_ready" not in st.session_state:

        with st.spinner("Processing PDF..."):

            pdf_hash = backend.get_pdf_hash(pdf)

            chunks = backend.process_pdf(pdf, pdf_hash)

            texts, tables, images = backend.separate_elements(chunks)

            st.write("Text chunks:", len(texts))
            st.write("Tables:", len(tables))
            st.write("Images:", len(images))

            text_summaries, table_summaries = backend.create_text_table_summaries(
                texts, tables
            )

            image_summaries = backend.create_image_summaries(images)

            vectorstore, docstore = backend.build_vectorstore(
                texts,
                tables,
                images,
                text_summaries,
                table_summaries,
                image_summaries
            )

            retriever = backend.MultiModalRetriever(
                vectorstore,
                docstore
            )

            chain = backend.build_chain(retriever)

            st.session_state.chain = chain
            st.session_state.rag_ready = True
            st.session_state.pdf_file = pdf

        st.success("PDF indexed successfully!")

    # --------------------------------
    # Chat Interface
    # --------------------------------

    if "rag_ready" in st.session_state:

        chat_col, source_col = st.columns([2, 1])

        # ==============================
        # LEFT SIDE → CHAT
        # ==============================

        with chat_col:

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:

                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            query = st.chat_input("Ask about the paper")

            if query:

                st.session_state.messages.append(
                    {"role": "user", "content": query}
                )

                with st.chat_message("user"):
                    st.markdown(query)

                result = st.session_state.chain.invoke(query)

                answer = result["response"]
                citations = result["citations"]
                context = result["context"]

                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                st.session_state.last_sources = citations
                st.session_state.last_context = context

        # ==============================
        # RIGHT SIDE → SOURCES
        # ==============================

        with source_col:

            if "last_sources" in st.session_state:

                citations = st.session_state.last_sources
                context = st.session_state.last_context

                st.subheader("📚 Sources")

                if citations["pages"]:
                    for p in citations["pages"]:
                        st.write("📄 Page", p)

                if citations["tables"]:
                    for t in citations["tables"]:
                        st.write("📊 Table Page", t)

                if citations["figures"]:
                    for f in citations["figures"]:
                        st.write("🖼 Figure Page", f)

                st.divider()

                # --------------------------
                # Images
                # --------------------------

                if context["image"]:

                    st.subheader("🖼 Figures")

                    for img in context["image"]:
                        display_image_base64(img)

                # --------------------------
                # Tables
                # --------------------------

                st.subheader("📊 Tables")

                for doc in context["text"]:
                    if hasattr(doc, "metadata"):

                        for el in doc.metadata.orig_elements or []:

                            if "Table" in str(type(el)):
                                display_table(el)

                # --------------------------
                # PDF Viewer
                # --------------------------

                if citations["pages"]:

                    st.subheader("📄 PDF Page")

                    page = st.selectbox(
                        "View page",
                        citations["pages"]
                    )

                    show_pdf_page(st.session_state.pdf_file, page)


if __name__ == "__main__":
    main()