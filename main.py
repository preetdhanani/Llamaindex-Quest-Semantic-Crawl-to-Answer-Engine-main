from web_crawl import web_crawler
from embedding import Embedder
from chunking import Chunker
from faiss_help import Faisss
import numpy as np
import pickle
import os
from grq import GroqAnswering

url = "https://docs.llamaindex.ai/en/stable/"
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

# storing Vectors
faiss = Faisss(embeddings,urls)
faiss = faiss.toFaiss()


# Example query 
query = input("Enter Text:")
query = chunker.chunk_question(query)
# Convert query to embedding
query_embedding = np.array(embedder.create_embeddings(query), dtype=np.float32)

distance, indices = faiss.search(query_embedding,2)
print("Nearest Neighbors' Indices:", indices)
print("Distances:", distance)

chunksss= pickle.load(open(f"static/{urls}_chunk_store.pkl", "rb"))

# for idx in indices[0]:  # FAISS returns 2D array, so take first list
#     print(f"Retrieved Text: {chunksss[idx]}")
# Retrieve actual text chunks based on indices
max_chunks =3
retrieved_chunks = [chunksss[idx] for idx in indices[0][:max_chunks]]  # FAISS returns 2D array, so take first list
# print(retrieved_chunks)

# Combine chunks into a single context string
context = "\n\n".join(doc.page_content[:1900] for doc in retrieved_chunks)

responce = GroqAnswering(context,query)

print(responce.groq_answering())

