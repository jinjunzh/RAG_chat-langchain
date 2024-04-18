"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
#from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OllamaEmbeddings(model="llama2")
#    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    DATABASE_HOST = os.environ.get("DATABASE_HOST", "127.0.0.1")
    DATABASE_PORT = os.environ.get("DATABASE_PORT", "5432")
    DATABASE_USERNAME = os.environ.get("DATABASE_USERNAME", "postgres")
    DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "test")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "jsp_database")
    RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "jsp_collection")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory='/mnt/home0/jasper/rag/chroma_data'
    )
    record_manager = SQLRecordManager(
        f"weaviate/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
    )

    record_manager.create_schema()
    print("ZJJDBG: TAG0\n")


#    docs_from_documentation = load_langchain_docs()
#    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
#    docs_from_api = load_api_docs()
#    logger.info(f"Loaded {len(docs_from_api)} docs from API")
#    docs_from_langsmith = load_langsmith_docs()
#    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")
#    test_docs1 = WebBaseLoader("https://docs.smith.langchain.com/user_guide").load()
#    test_docs2 = WebBaseLoader("https://clickhouse.com/blog/lz4-compression-in-clickhouse").load()

    test_docs1 = WebBaseLoader("https://clickhouse.com/blog/optimize-clickhouse-codecs-compression-schema").load()
    test_docs2 = WebBaseLoader("https://clickhouse.com/docs/en/development/building_and_benchmarking_deflate_qpl").load() 
    test_docs3 = WebBaseLoader("https://clickhouse.com/blog/lz4-compression-in-clickhouse").load()   

    docs_transformed = text_splitter.split_documents(test_docs1+test_docs2+test_docs3)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    #num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    #logger.info(
    #    f"LangChain now has this many vectors: {num_vecs}",
    #)




###########Validate ingestion start#######
#    llm = Ollama(model="zephyr-local")
#    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
#    <context>
#    {context}
#    </context>
#    Question: {input}""")
#    document_chain = create_stuff_documents_chain(llm, prompt)

#    retriever = vectorstore.as_retriever()
#    retrieval_chain = create_retrieval_chain(retriever, document_chain)
#    response = retrieval_chain.invoke({"input": "What is Multi-instance?"})
#    print(response["answer"])
###########Validate ingestion end#######




if __name__ == "__main__":
    ingest_docs()
