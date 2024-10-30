from langchain.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def create_embeddings(self, chunks):
        return [self.embeddings.embed_query(chunk.page_content) for chunk in chunks]
