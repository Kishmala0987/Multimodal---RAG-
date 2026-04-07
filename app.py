import os
import base64
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import backend


# ==============================
# HELPERS
# ==============================

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
    st.caption(f"Page {page_number}")
    st.text(text[:2000])


# ==============================
# MAIN APP
# ==============================

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

    # ==============================
    # PROCESS PDF
    # ==============================

    if pdf and "rag_ready" not in st.session_state:
        with st.spinner("Processing PDF..."):
            pdf_hash = backend.get_pdf_hash(pdf)
            chunks = backend.process_pdf(pdf, pdf_hash)
            texts, tables, images = backend.separate_elements(chunks)

            st.write(f"Text chunks: {len(texts)} | Tables: {len(tables)} | Images: {len(images)}")

            text_summaries, table_summaries = backend.create_text_table_summaries(texts, tables)
            image_summaries = backend.create_image_summaries(images)

            vectorstore, docstore = backend.build_vectorstore(
                texts, tables, images,
                text_summaries, table_summaries, image_summaries
            )

            retriever = backend.MultiModalRetriever(vectorstore, docstore)
            chain = backend.build_chain(retriever)

            st.session_state.chain = chain
            st.session_state.rag_ready = True
            st.session_state.pdf_file = pdf

        st.success("PDF indexed successfully!")

    # ==============================
    # CHAT — FULL WIDTH
    # ==============================

    if "rag_ready" in st.session_state:

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("Ask about the paper")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke(query)

            answer   = result["response"]
            citations = result["citations"]
            context  = result["context"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_citations = citations
            st.session_state.last_context   = context

        # ==============================
        # SOURCES — ACCORDION BELOW CHAT
        # ==============================

        if "last_citations" in st.session_state:
            citations = st.session_state.last_citations
            context   = st.session_state.last_context

            st.divider()
            st.subheader("📚 Sources")

            # ---------- Page Citations ----------
            with st.expander("📄 Page Citations", expanded=True):
                if citations["pages"]:
                    badges = " · ".join(f"Page {p}" for p in citations["pages"])
                    st.markdown(f"`{badges}`")
                else:
                    st.write("No page citations found.")

                if citations["tables"]:
                    st.caption("Tables on pages: " + ", ".join(str(t) for t in citations["tables"]))

                if citations["figures"]:
                    st.caption("Figures on pages: " + ", ".join(str(f) for f in citations["figures"]))

            # ---------- Figures ----------
            with st.expander("🖼️ Figures", expanded=False):
                if context.get("image"):
                    cols = st.columns(2)
                    for i, img in enumerate(context["image"]):
                        with cols[i % 2]:
                            display_image_base64(img)
                else:
                    st.write("No figures retrieved for this query.")

            # ---------- Tables ----------
            with st.expander("📊 Tables", expanded=False):
                found_table = False
                for doc in context.get("text", []):
                    if hasattr(doc, "metadata"):
                        for el in doc.metadata.orig_elements or []:
                            if "Table" in str(type(el)):
                                display_table(el)
                                st.divider()
                                found_table = True
                if not found_table:
                    st.write("No tables retrieved for this query.")




if __name__ == "__main__":
    main()