# llamaindex Documentation QA System

This project implements a Question Answering (QA) system for llamaindex documentation. It crawls the llamaindex documentation, processes the data, stores it in a vector database, and uses advanced retrieval techniques to answer user queries.

## Features

- Web crawling of llamaindex documentation
- Advanced data chunking based on semantic similarity
- Vector embedding creation and storage in Faiss database
- Query expansion for improved retrieval
- Hybrid retrieval combining BM25 and BERT-based methods
- Question answering using a Language Model

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository:
2. Create a virtual environment (optional but recommended):
3. Install the required dependencies:

### Dependencies

The main dependencies for this project are:

- scrapy: For web crawling
- sentence-transformers: For text embeddings
- nltk: For natural language processing tasks
- torch and transformers: For working with transformer models
- selenium and webdriver_manager: For web scraping
- Faiss: For interacting with vector database

For a complete list of dependencies, refer to the `requirements.txt` file.

## Running the System

1. Run the main script:
   
2. The system will start by crawling the llamaindex documentation, processing the data, and storing it in the faiss database. This initial setup may take some time.

3. Once the setup is complete, you can start asking questions about Llamaindex. The system will provide answers based on the retrieved information.

5. To exit the system, type 'quit' when prompted for a question.

## Project Structure

- `main.py`: The main script that orchestrates the entire process.
- `web_crawler.py`: Contains the web crawling logic.
- `chunking.py`: Implements advanced data chunking techniques and implements query expansion techniques.
- `embedding.py`: Handles the creation of vector embeddings.
- `faiss_help.py`: Manages interactions with the Milvus database.
- `grq.py`: Manages the question answering process using a language model. 
- `retrieval/hybrid_retrieval.py`: Contains the hybrid retrieval logic.

## Customization

- You can adjust the embedding model by modifying the `SentenceTransformer` model in `main.py`.
- The depth of web crawling can be adjusted in the `web_crawl` function (currently set to 5 levels).
- The number of retrieved chunks for answering can be modified by changing the `top_k` parameter in the `faiss_help` method call.(currently set to 2 levels)

## Troubleshooting

If you encounter any issues:
- Ensure all dependencies are correctly installed.
- Verify that you have a stable internet connection for web crawling and model downloads.

For any persistent problems, please open an issue in the GitHub repository.

