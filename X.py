from web_crawl import web_crawler
from embedding import Embedder
from chunking import Chunker
from milvus import milvus

url = "https://docs.llamaindex.ai/en/stable/"


craw = web_crawler(url)
craw.crawl() 
 
# Data Chunking
embedder = Embedder()
chunker = Chunker(embedder.embeddings)
chunks = chunker.chunk_text(craw.all_text)
all_text_file = "all_text_content.txt"
craw.save_all_text(all_text_file)

# Embedding Creation
embeddings = embedder.create_embeddings(chunks)

# storing Vectors

milvus_db = milvus()
collection = milvus_db.create_collection("cuda",dim=384)
milvus_db.insert_data(collection,embeddings,chunks)



   