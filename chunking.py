from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

class Chunker:
    def __init__(self, embeddings):
        self.text_splitter = SemanticChunker(
            embeddings, breakpoint_threshold_type="percentile"
        )

    def chunk_text(self, all_text):
        docs = []
        chunk_file = "chunks_in_progress.txt"
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for i, (text, url) in enumerate(all_text):
                chunks = self.text_splitter.create_documents([text])
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk.page_content,
                        metadata={
                            'source': url,
                        }
                    )
                    docs.append(doc)
                    
                    f.write(f"Document {i+1}, Chunk {j+1}:\n")
                    f.write(f"Source URL: {url}\n")
                    f.write(chunk.page_content)
                    f.write("\n\n" + "="*50 + "\n\n")
        
        return docs