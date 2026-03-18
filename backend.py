import os
import uuid
import base64
import tempfile
import time
import hashlib

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_community.vectorstores import Chroma

load_dotenv()

# ==============================
# ENVIRONMENT
# ==============================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# ==============================
# MODELS
# ==============================

summary_model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
vision_model = ChatOpenAI(
    model="gpt-4o-mini"
)
parser = StrOutputParser()


# ==============================
# HELPERS
# ==============================

def get_pdf_hash(uploaded_file):
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()


# ==============================
# 1️⃣ PDF PROCESSING
# ==============================

def process_pdf(uploaded_file, pdf_hash: str):
    """
    pdf_hash is accepted so app.py can pass it in,
    but actual Streamlit caching lives in app.py via st.cache_data.
    We keep processing pure here.
    """
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=5000,
            combine_text_under_n_chars=1000,
            new_after_n_chars=3000,
        )
    finally:
        os.unlink(file_path)

    return chunks


# ==============================
# 2️⃣ SEPARATE ELEMENT TYPES
# ==============================

def separate_elements(chunks):
    tables = []
    texts = []
    images = []

    for chunk in chunks:
        texts.append(chunk)
        if chunk.metadata.orig_elements:
            for el in chunk.metadata.orig_elements:
                if "Table" in str(type(el)):
                    tables.append(el)
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)

    return texts, tables, images


# ==============================
# 3️⃣ TEXT + TABLE SUMMARIES  (rate-limit safe)
# ==============================

summary_prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that summarizes tables and text.

If the input is a table summarize the key insights.
If the input is text summarize the main idea concisely.

Respond only with the summary.

Content:
{element}
"""
)

summary_chain = summary_prompt | summary_model | parser


def _batched_invoke(items, batch_size: int = 5, delay: float = 3.0):
    """
    Send items to Groq in small batches with a pause in between
    to respect the free-tier RPM limit.
    batch_size=5 and delay=3s keeps you well under 30 RPM.
    Tune these values if you upgrade your Groq tier.
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(items), batch_size)):
        batch = items[start : start + batch_size]
        results.extend(summary_chain.batch(batch))

        # Sleep after every batch except the last one
        if i < total_batches - 1:
            time.sleep(delay)

    return results


def create_text_table_summaries(texts, tables):
    text_summaries = _batched_invoke(texts, batch_size=1, delay=5.0)

    tables_html = [t.metadata.text_as_html for t in tables]
    table_summaries = _batched_invoke(tables_html, batch_size=1, delay=5.0) if tables_html else []

    return text_summaries, table_summaries


# ==============================
# 4️⃣ IMAGE SUMMARIES
# ==============================

def create_image_summaries(images):
    if not images:
        return []
    prompt_template = """
    You are an AI assistant that summarizes images of charts or tables.
    - Describe the main message of the image.
    - Identify the type of chart (bar, line, pie, etc.).
    - Keep the summary concise, clear, and factual.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
            ],
        }
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | vision_model | parser
    image_summaries = chain.batch(images)
    return image_summaries


# ==============================
# 5️⃣ BUILD VECTORSTORE
# ==============================

def build_vectorstore(texts, tables, images, text_summaries, table_summaries, image_summaries):
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(),
    )

    docstore = InMemoryStore()
    id_key = "doc_id"

    # TEXT
    text_ids = [str(uuid.uuid4()) for _ in texts]
    vectorstore.add_documents(
        [
            Document(page_content=s, metadata={id_key: text_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
    )
    docstore.mset(list(zip(text_ids, texts)))

    # TABLE
    if table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        vectorstore.add_documents(
            [
                Document(page_content=s, metadata={id_key: table_ids[i]})
                for i, s in enumerate(table_summaries)
            ]
        )
        docstore.mset(list(zip(table_ids, tables)))

    # IMAGE
    if image_summaries:
        image_ids = [str(uuid.uuid4()) for _ in images]
        vectorstore.add_documents(
            [
                Document(page_content=s, metadata={id_key: image_ids[i]})
                for i, s in enumerate(image_summaries)
            ]
        )
        docstore.mset(list(zip(image_ids, images)))

    return vectorstore, docstore


# ==============================
# 6️⃣ RETRIEVER
# ==============================

class MultiModalRetriever:
    def __init__(self, vectorstore, docstore, id_key="doc_id"):
        self.vectorstore = vectorstore.as_retriever()
        self.docstore = docstore
        self.id_key = id_key

    def get_relevant_documents(self, query):
        hits = self.vectorstore.invoke(query)
        ids = [hit.metadata[self.id_key] for hit in hits]
        docs = self.docstore.mget(ids)
        return docs


# ==============================
# 7️⃣ PARSE RETRIEVED DOCS
# ==============================

def parse_docs(docs):
    images, texts = [], []
    for doc in docs:
        try:
            base64.b64decode(doc, validate=True)
            images.append(doc)
        except Exception:
            texts.append(doc)
    return {"image": images, "text": texts, "raw_docs": docs}


# ==============================
# 8️⃣ BUILD FINAL PROMPT
# ==============================

def build_prompt(kwargs):
    docs = kwargs["context"]
    context_text = ""

    for t in docs.get("text", []):
        if hasattr(t, "text"):
            context_text += t.text + "\n"
        elif isinstance(t, str):
            context_text += t + "\n"

    prompt_template = """
Answer the question based only on the following context, which may include text, tables, and image summaries.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question:
{question}

After the answer list the source pages like:

Sources:
Page X
Page Y
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt.format(context=context_text, question=kwargs["question"])


def extract_citations(docs):
    pages, figures, tables = set(), [], []

    for doc in docs:
        if hasattr(doc, "metadata") and hasattr(doc.metadata, "orig_elements"):
            for el in doc.metadata.orig_elements:
                if hasattr(el.metadata, "page_number"):
                    pages.add(el.metadata.page_number)
                if "Image" in str(type(el)):
                    figures.append(el.metadata.page_number)
                if "Table" in str(type(el)):
                    tables.append(el.metadata.page_number)

    return {"pages": sorted(list(pages)), "figures": figures, "tables": tables}


# ==============================
# 9️⃣ BUILD QA CHAIN
# ==============================

def build_chain(retriever):
    qa_model = ChatOpenAI(model="gpt-4o-mini")

    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x))
            | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=RunnableLambda(build_prompt) | qa_model | parser,
            citations=RunnableLambda(
                lambda x: extract_citations(x["context"]["raw_docs"])
            ),
        context=RunnableLambda(lambda x: x["context"])
        )
    )

    return chain