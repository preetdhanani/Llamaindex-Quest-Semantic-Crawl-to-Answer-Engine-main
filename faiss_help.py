import faiss
import os
import numpy as np

class Faisss:
    def __init__(self,embeddings):
        self.embedding_dims=384
        self.embeddings = np.array(embeddings,dtype=np.float32)
        
        

    def toFaiss(self):
        if os.path.exists("static/faiss_index.bin"):
            self.index = faiss.read_index("static/faiss_index.bin")
            print("DB load Successfull")
        else:
            print("///////////////////")
            print(self.embeddings.shape)
            print("DB Creatting")
            self.index = faiss.IndexFlatL2(self.embedding_dims)
            self.index.add(self.embeddings)
            faiss.write_index(self.index, "static/faiss_index.bin")
            print("DB saved")
        return self.index