
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import configparser

class milvus:
    def __init__(self):
        cfp = configparser.RawConfigParser()
        cfp.read('config.ini')
        milvus_uri = cfp.get('example', 'uri')
        token = cfp.get('example', 'token')
        connections.connect("default", uri=milvus_uri, token=token)
    
    def create_collection(self, collection_name, dim):
        # Define the fields for the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_increment=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=3000)  # Corrected field definition
        ]
        
        # Create schema for the collection
        schema = CollectionSchema(fields=fields, description="Collection for embeddings")

        # Create the collection if it doesn't exist
        if collection_name not in utility.list_collections():
            collection = Collection(name=collection_name, schema=schema)
        else:
            collection = Collection(name=collection_name)
        
        return collection

    def insert_data(self, collection, embeddings, chunks,):
        num_embeddings = len(embeddings)

        # Ensure that embeddings and chunks have the same length
        if num_embeddings != len(chunks):
            raise ValueError("Number of embeddings and text chunks must be the same.")
        
        # Extract text content from Document objects in chunks and truncate if necessary
        text_chunks = [
            chunk.page_content[:1000] for chunk in chunks
        ]

        # Prepare data for insertion (in column-wise format)
        entities = [
            [i for i in range(num_embeddings)],  # IDs (auto-generated in this case)
            embeddings,  # Embedding vectors
            text_chunks  # Corresponding text chunks (as strings, truncated if needed)
        ]

        # Insert data into the collection
        collection.insert(entities)
        print(f"Inserted {num_embeddings} embeddings into Milvus collection '{collection.name}'.")
