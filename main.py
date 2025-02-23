from web_crawl import web_crawler
from embedding import Embedder
from chunking import Chunker
from hybrid_retrieval import HybridSearch
import numpy as np
import pickle
import os
from grq import GroqAnswering

url = "https://docs.llamaindex.ai/"
urls = url.split('/')[2]
 
craw = web_crawler(url)
craw.crawl() 


# Data Chunking
embedder = Embedder()
chunker = Chunker(embedder.embeddings,url)
chunks = chunker.chunk_text(craw.all_text)


# Embedding Creation
if not os.path.exists(f"static/{urls}_embeddings.pkl"):
    embeddings = embedder.create_embeddings(chunks)
    pickle.dump(embeddings, open(f"static/{urls}_embeddings.pkl", "wb"))
else:
    embeddings = pickle.load(open(f"static/{urls}_embeddings.pkl", "rb"))
    print("embeddings loaded")
print("embaddings done")


hybrid_faiss = HybridSearch(embeddings, chunks, urls )


while True:

    query = input("Enter Text:")
    if query.lower() == 'quit':
        break
    else:
        query = chunker.chunk_question(query)
        # Convert query to embedding
        query_embedding = np.array(embedder.create_embeddings(query), dtype=np.float32)

        context = hybrid_faiss.search(query_embedding, query, k=2, alpha=0.7)

        # print(context)
        # Retrieve actual text chunks based on indices
        max_chunks =3

        # Combine chunks into a single context string
        context = context[:max_chunks]
        context = "\n\n".join(doc.page_content[:2000] for doc in context)

        responce = GroqAnswering(context,query)

        print(responce.groq_answering())

