from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
import os
import pickle
from tqdm import tqdm
class Chunker:
    def __init__(self, embeddings,start_url):
        self.text_splitter = SemanticChunker(
            embeddings, breakpoint_threshold_type="percentile"
        )
        self.url = start_url.split("/")[2]

    def chunk_text(self, all_text):
        docs = []
        self.chunk_store = []
        chunk_file = f"static/{self.url}_chunks_in_progress.txt"
        if not os.path.exists(f"static/{self.url}_chunks_in_progress.txt"):
              
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for i, (text, url) in enumerate(tqdm(all_text, desc="Chunking Progress", total=len(all_text))):

                    chunks = self.text_splitter.create_documents([text])
                    for j, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk.page_content,
                            metadata={
                                'source': url,
                            }
                        )
                        docs.append(doc)
                        self.chunk_store.append(doc)
                        
                        f.write(f"Document {i+1}, Chunk {j+1}:\n")
                        f.write(f"Source URL: {url}\n")
                        f.write(chunk.page_content)
                        f.write("\n\n" + "="*50 + "\n\n")
                    
                    pickle.dump(self.chunk_store, open(f"static/{self.url}_chunk_store.pkl", "wb"))
            
            return docs
        else:
            self.chunk_store = pickle.load(open(f"static/{self.url}_chunk_store.pkl", "rb"))
            print("chunks existed")
            return self.chunk_store

    def chunk_question(self, query):
        docs = []
        print(query)
        chunks = self.text_splitter.create_documents([query])
        print(chunks)
        for chunk in chunks:
            doc = Document(page_content=chunk.page_content,)

            docs.append(doc)
        print(docs)
        return docs

