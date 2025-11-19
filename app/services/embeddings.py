# This file contains the code for generating embeddings.
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import Tag
from typing import Optional
import os
import time
from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import HTMLSemanticPreservingSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from ..utils.web_loader import WebLoader
from google.api_core.exceptions import ResourceExhausted
# from langchain_google_genai._common import GoogleGenerativeAIError
# from langsmith import traceable
load_dotenv()  # Load .env file

class TravelEmbeddingPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embed_model: str = "nomic-embed-text", # "gemini-1.5-embed-text" or "nomic-embed-text"
        use_embeddings: bool = True,
    ):
        self.loader = WebLoader()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
            ],  # no empty string to avoid regex issues
        )
        self.embed_model = embed_model
        self.use_embeddings = use_embeddings
        self._embedder: Optional[OllamaEmbeddings] = None

    def _ensure_embedder(self):
        if not self.use_embeddings:
            return
        # if not os.getenv("GOOGLE_API_KEY"):
        #     raise RuntimeError("GOOGLE_API_KEY not set. Add it to your .env.")
        # if self._embedder is None:
        #     self._embedder = GoogleGenerativeAIEmbeddings(model=self.embed_model)
        if self._embedder is None: 
            self._embedder = OllamaEmbeddings(model=self.embed_model)

    def load_docs(self):
        return self.loader.load()

    def split_docs(self, docs):
        return self.splitter.split_documents(docs)

    def to_texts_and_meta(self, chunks) -> tuple[list[str], list[dict]]:
        return [c.page_content for c in chunks], [dict(c.metadata) for c in chunks]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 78,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self._ensure_embedder()
        if not self._embedder:
            return []
        
        vectors: list[list[float]] = []
        i = 0
        while i < len(texts):
            batch = texts[i : i + batch_size]
            attempt = 0
            while True:
                try:  
                    vectors.extend(self._embedder.embed_documents(batch))
                    break
                except ( ResourceExhausted) as e:
                    attempt += 1
                    if attempt >  max_retries:
                        raise
                    sleep_s = base_delay * (2 ** (attempt - 1))
                    if attempt >= 5 and batch_size > 10:
                        batch_size = max(10, batch_size // 2)
                        batch = texts[i : i + batch_size]
                    time.sleep(sleep_s)
                except Exception:
                    raise
                
            i += batch_size
        return vectors
                    
    def run(self):
        docs = self.load_docs()
        chunks = self.split_docs(docs)
        texts,metadatas = self.to_texts_and_meta(chunks)
        embeddings = self.embed_texts(texts) if self.use_embeddings else None
        return {
            "docs": docs,
            "chunks": chunks,
            "texts": texts,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }


# if __name__ == "__main__":
#     pipeline = TravelEmbeddingPipeline(use_embeddings=True)  # set True if GOOGLE_API_KEY is configured
#     out = pipeline.run()
#     print(f"docs={len(out['docs'])}, chunks={len(out['chunks'])}")
#     if out["texts"]:
#         print(f"first chunk chars={len(out['texts'])}, words={len(out['texts'])}")
#     if out["embeddings"] is not None:
#         print(f"embeddings computed: {len(out['embeddings'])}")


    # out = TravelEmbeddingPipeline().run()
    # docs, chunks, texts = out["docs"], out["chunks"], out["texts"]

    # print(f"docs={len(docs)}, chunks={len(chunks)}, texts={len(texts)}")
    # if docs:
    #     d0 = docs[0]
    #     print(f"doc[0] source={d0.metadata.get('source')}, chars={len(d0.page_content)}, words={len(d0.page_content.split())}")
    # if texts:
    #     t0 = texts[0]
    #     print(f"first chunk chars={len(t0)}, words={len(t0.split())}")
    #     print(f"preview: {t0[:1000]}...")
    # if chunks: 
    #     print(chunks[101])

# headers_to_split_on = [
#     ("h1", "Header 1"),
#     ("h2", "Header 2"),
# ]

# def code_handler(element: Tag) -> str:
#     data_lang = element.get("data-lang")
#     code_format = f"<code:{data_lang}>{element.get_text()}</code>"

#     return code_format


# splitter = HTMLSemanticPreservingSplitter(
#     headers_to_split_on=headers_to_split_on,
#     separators=["\n\n", "\n", ". ", "! ", "? "],
#     max_chunk_size=50,
#     preserve_images=True,
#     preserve_videos=True,
#     elements_to_preserve=["table", "ul", "ol", "code"],
#     denylist_tags=["script", "style", "head"],
#     custom_handlers={"code": code_handler},
# )

# documents = splitter.split_text()