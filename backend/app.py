import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import json
import numpy as np
from openai import OpenAI
import cohere
import time
import torch
import open_clip

DATA_PATH = os.environ["DATA_PATH"]
FAISS_TEXT_VECTOR_DB_PATH = os.environ["FAISS_TEXT_VECTOR_DB_PATH"]
FAISS_IMAGE_VECTOR_DB_PATH = os.environ["FAISS_IMAGE_VECTOR_DB_PATH"]
AMAZON_FASHION_FILE = os.environ["AMAZON_FASHION_FILE"]
OPENAI_EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
NUMBER_OF_RETRIEVED_ITEMS = 100
NUMBER_OF_OUTPUT_ITEMS = 10

# FastAPI app initialization
app = FastAPI()

# Allowing communication with the frontend
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body for the search endpoint
class QueryRequest(BaseModel):
    query: str


def load_vector_db(path: str) -> faiss.Index:
    """
    Load the FAISS vector database from the specified path.
    This FAISS index was precomputed using the embeddings of 100K items from the Amazon Fashion dataset.
    For more details, refer to the README file in the repository.
    """
    print(f"Loading FAISS vector DB from {path}")
    index = faiss.read_index(path)
    print("FAISS vector DB loaded")
    return index

def load_CLIP_model() -> open_clip.model:
    """
    Load the CLIP model for text embedding to compare with images.
    This same model is used to compute the embeddings of the images in the dataset.
    """
    print(f"Loading CLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to("cpu").eval()
    print("CLIP model loaded")
    return model

def embed_query(openai_client: OpenAI, model: open_clip.model, query: str, mode: str) -> np.ndarray:
    """
    Embed the user's query using OpenAI's text embedding model 
    or OpenAI's CLIP model depemnding on the mode (text or image).
    """
    print("Start embedding the query...")
    if mode == "text":
        response = openai_client.embeddings.create(
            input=query,
            model=OPENAI_EMBEDDING_MODEL
        )
        # Get embeddings from the response
        embeds = np.array(response.data[0].embedding).reshape(1, -1)
    elif mode == "image":
        with torch.no_grad():
            text_tokens = open_clip.tokenize([query]).to("cpu")
            embeds = model.encode_text(text_tokens).cpu().numpy().astype('float32')
            embeds /= np.linalg.norm(embeds)  # normalize for cosine similarity
    print(f"Query embedded in MODE : {mode}")
    return embeds

def retrieve_indices(index: faiss.Index, query_vector: np.ndarray, k: int) -> list:
    """
    Retrieve the k nearest neighbors of the input query vector from the FAISS index.
    The FAISS index uses L2 distance for similarity search.
    """
    # Search for the k nearest neighbors in the FAISS index
    # The query_vector should be of shape (1, d) where d is the dimension of the embeddings
    print("Retrieving similar items...")
    distances, indices = index.search(query_vector, k)
    print(f"Retrieved {len(indices.tolist()[0])} similar items")
    return indices.tolist()[0]

def format_title(title: str) -> str:
    """
    Format the title of the Amazon item so that it is nice in the UI.
    """
    # Remove unwanted characters
    title = title.replace('\n', ' ').replace('\r', '').replace('\t', '').strip()
    # Limit the size of the title to 50 characters
    # If longer than 50 characters, truncate it and add "..."
    if len(title) > 50:
        title = title[:47] + "..."
    return title

def get_items_from_indices(indices: list) -> dict:
    """
    Get the items from the jsonl file using a list of indices.
    The jsonl file contains the Amazon Fashion dataset.
    Each line is a JSON object representing an item.
    """
    # Read the line of jsonl file corresponding to the indices
    print("Reading the retrieved items from the jsonl file...")
    items = {}
    # Extract the lines from the jsonl file using the indices
    with open(AMAZON_FASHION_FILE, 'r') as f:
        lines = f.readlines()
        extracted_lines = [lines[i] for i in indices]
        f.close()
    # Parse the json lines and extract the relevant fields
    for local_index, db_index in enumerate(indices):
        item = json.loads(extracted_lines[local_index])
        # Extract the relevant fields from the JSON object
        formatted_title = format_title(item.get("title", ""))
        # Concatenate the title, description, and features to create the content
        # The same method was used to create the embeddings in the FAISS index
        content = item.get("title", "") + " " + ' '.join(item.get("description", [])) + ' '.join(item.get("features", []))
        # Select first image if available, else use a placeholder image
        image_url = item.get("images", [{'large':'https://upload.wikimedia.org/wikipedia/commons/0/0a/No-image-available.png'}])[0]['large']
        # Create a dictionary entry with the relevant fields
        items[db_index] = {
            "text":content,
            "image_url": image_url,
            "title": formatted_title,
            "id": item.get("parent_asin", ""),
        }
    print("Retrieving done.")
    return items 

def rerank_items(cohere_client: cohere.Client, top_n: int, query: str, indices: list) -> list:
    """
    Rerank the items using Cohere's reranking model.
    The reranking model uses the query and the items to compute a relevance score for each item.
    The items are then sorted based on the relevance score.
    """
    items = get_items_from_indices(indices)
    # Rerank the items using Cohere
    print("Reranking items...")
    response = cohere_client.rerank(
        query=query,
        documents=[items[idx]["text"] for idx in items],
        top_n=top_n,
        model="rerank-v3.5",
    )
    print("Reranking done.")
    reranked_idx = []
    for result in response.results:
        reranked_idx.append(indices[result.index])
    print("Reranking postprocessed.")
    return reranked_idx


@app.on_event("startup")
async def initialize():
    """
    Initialize the FastAPI app.
    Load the CLIP model, the 2 FAISS vector databases and initialize the OpenAI and Cohere clients.
    """
    text_vector_db = load_vector_db(FAISS_TEXT_VECTOR_DB_PATH)
    image_vector_db = load_vector_db(FAISS_IMAGE_VECTOR_DB_PATH)
    clip_model = load_CLIP_model()
    openai_client = OpenAI()
    cohere_client = cohere.ClientV2(COHERE_API_KEY)
    # Store in the app's state
    app.state.image_vector_db = image_vector_db
    app.state.text_vector_db = text_vector_db
    app.state.clip_model = clip_model
    app.state.openai_client = openai_client
    app.state.cohere_client = cohere_client


# Define the endpoint for searching the text defining items
@app.post("/search")
async def search_items(request: QueryRequest):
    """
    Search for items based on the user's query.
    Perform the following steps:
    1. Embed the query using OpenAI's embedding model.
    2. Retrieve the top k items from the FAISS Text Embedding vector database.
    3. Rerank the items using Cohere's reranking model.
    4. Get the items from the indices.
    5. Return the items to the user.
    """
    try:
        start_time = time.time()
        query = request.query
        print(f"STEP 1: Received query: {query}")
        print("==========================")
        print("==========================")
        print(f"STEP 2 - Start: Embedding query: {query}")
        embedded_query = embed_query(app.state.openai_client, app.state.clip_model, query, "text")
        print(f"STEP 2 - End: Query embedded in the following vector: {embedded_query}")
        print("==========================")
        print("==========================")
        print(f"STEP 3 - Start: Retrieving from FAISS index of size {app.state.text_vector_db.ntotal}")
        indices = retrieve_indices(app.state.text_vector_db, embedded_query, NUMBER_OF_RETRIEVED_ITEMS)
        print(f"STEP 3 - End: Retrieved {NUMBER_OF_RETRIEVED_ITEMS} items")
        print("==========================")
        print("==========================")
        print(f"STEP 4 - Start: Reranking the {NUMBER_OF_RETRIEVED_ITEMS} retrieved items")
        reranked_indices = rerank_items(app.state.cohere_client, NUMBER_OF_OUTPUT_ITEMS, query, indices)
        print(f"STEP 4 - End: Reranked the items and outputted the top {NUMBER_OF_OUTPUT_ITEMS} items")
        print("==========================")
        print("==========================")
        print(f"STEP 5 - Start: Extract attributes of the {NUMBER_OF_OUTPUT_ITEMS} from JSONL file")
        items = get_items_from_indices(reranked_indices)
        print(f"STEP 5 - End: Extracted the attributes of the items")
        print(f"List of ordered Amazon IDs: {[items[idx]['id'] for idx in reranked_indices]}")
        print("==========================")
        print("==========================")
        print("TOTAL PROCESS TIME: {:.2f} seconds".format(time.time() - start_time))
        return [items[idx] for idx in reranked_indices]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the endpoint for searching images of items specifically
@app.post("/image-search")
async def search_items(request: QueryRequest):
    """
    Search for items based on the user's query.
    Perform the following steps:
    1. Embed the query using OpenAI's embedding model.
    2. Retrieve the top N images from the FAISS Image Embedding vector database.
    4. Get the items from the indices.
    5. Return the items to the user.
    """
    try:
        start_time = time.time()
        query = request.query
        print(f"STEP 1: Received query: {query}")
        print("==========================")
        print("==========================")
        print(f"STEP 2 - Start: Embedding query: {query}")
        embedded_query = embed_query(app.state.openai_client, app.state.clip_model, query, "image")
        print(f"STEP 2 - End: Query embedded in the following vector: {embedded_query}")
        print("==========================")
        print("==========================")
        print(f"STEP 3 - Start: Retrieving from FAISS index of size {app.state.image_vector_db.ntotal}")
        ranked_indices = retrieve_indices(app.state.image_vector_db, embedded_query, NUMBER_OF_OUTPUT_ITEMS)
        print(f"STEP 3 - End: Retrieved {NUMBER_OF_OUTPUT_ITEMS} items")
        print("==========================")
        print("==========================")
        print(f"STEP 4 - Start: Extract attributes of the {NUMBER_OF_OUTPUT_ITEMS} from JSONL file")
        items = get_items_from_indices(ranked_indices)
        print(f"STEP 4 - End: Extracted the attributes of the items")
        print(f"List of ordered Amazon IDs: {[items[idx]['id'] for idx in ranked_indices]}")
        print("==========================")
        print("==========================")
        print("TOTAL PROCESS TIME: {:.2f} seconds".format(time.time() - start_time))
        return [items[idx] for idx in ranked_indices]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the server is running.
    Returns a simple JSON response indicating the server's status.
    """
    return {"status": "healthy"}
