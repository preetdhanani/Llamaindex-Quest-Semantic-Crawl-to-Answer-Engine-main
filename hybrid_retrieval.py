import faiss
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



class HybridSearch:
    def __init__(self,embeddings, documents, urls):
        """
        Initializes the hybrid retrieval system with FAISS for dense retrieval
        and BM25 for sparse retrieval.
        """
        self.embedding_dims=384
        self.embeddings = np.array(embeddings,dtype=np.float32)
        self.urls = urls
        self.documents = documents


        ##  Faiss's work 

        if os.path.exists(f"static/{self.urls}_faiss_index.bin"):
            self.index = faiss.read_index(f"static/{self.urls}_faiss_index.bin")
            print("DB load Successfull")
        else:
            print("///////////////////")
            print(self.embeddings.shape)
            print("DB Creatting")
            
            self.index = faiss.IndexFlatL2(self.embedding_dims)
            self.index.add(self.embeddings)
            
            faiss.write_index(self.index, f"static/{self.urls}_faiss_index.bin")
            print("DB saved")

        ## BM25's work 

        # Extract text from Document objects for BM25
        self.text_corpus = [doc.page_content for doc in self.documents] 

        tokenized_docs = [text.split() for text in self.text_corpus]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    
    def search(self, query_embadding, query, k=5, alpha=0.7):
        """
        Performs hybrid search using both FAISS and BM25.
        
        Args:
        - query_embedding : embadded query
        - query (str): The input query.
        - k (int): Number of results to return.
        - alpha (float): Weight for FAISS results (1-alpha for BM25).
        
        Returns:
        - List of top-k retrieved documents.
        """
        
        distance, indices = self.index.search(query_embadding, k)  # FAISS retrieval

        # Normalize FAISS scores (smaller distances = higher relevance)
        faiss_scores = 1 / (distance[0] + 1e-5)  # Avoid division by zero

        # Get BM25 (Sparse) Search Results
        query_corpus = [doc.page_content for doc in self.documents]
        
        bm25_scores = self.bm25.get_scores(query_corpus[0].split())
        bm25_top_k = np.argsort(bm25_scores)[-k:]  # Top-k BM25 indices

        # Normalize BM25 scores
        bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
        
        # Hybrid Scoring
        combined_scores = {}
        
        for idx, score in zip(indices[0], faiss_scores):  # FAISS results
            combined_scores[idx] = alpha * score

        for idx in bm25_top_k:  # BM25 results
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * bm25_scores[idx]
        
        # Sort results by hybrid score
        final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [self.documents[idx] for idx, _ in final_results[:k]]  # Return top-k documents


        
