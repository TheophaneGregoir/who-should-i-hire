import os
from pathlib import Path
import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
from openai import OpenAI
import cohere
import time
import pickle

DATA_PATH = os.environ["DATA_PATH"]
FAISS_TEXT_VECTOR_DB_PATH = os.environ["FAISS_TEXT_VECTOR_DB_PATH"]
CELEB_TO_TEXT_PATH = os.environ["CELEB_TO_TEXT_PATH"]
RESUME_FOLDER = os.environ["RESUME_FOLDER"]
OPENAI_EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
NUMBER_OF_RETRIEVED_ITEMS = 100
NUMBER_OF_OUTPUT_ITEMS = 5

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
    This FAISS index was precomputed using the embeddings of 800+ resume extracted with Mistral OCR.
    """
    print(f"Loading FAISS vector DB from {path}")
    index = faiss.read_index(path)
    print("FAISS vector DB loaded")
    return index

def load_dict(path: str) -> dict:
    """
    Load a dictionary from a pickle file.
    This dictionary maps celebrity names to their text descriptions.
    """
    print(f"Loading dictionary from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("Dictionary loaded")
    return data

def generate_qualities_and_skills(openai_client: OpenAI, query: str) -> str:
    """Use ``gpt-5-nano`` to derive 5 qualities and 5 skills from the query."""

    print("Generating qualities and skills with gpt-5-nano...")
    prompt = (
        "You are given a candidate profile description.\n" +
        query +
        "\n\nRespond strictly in the following format with no extra text:\n" +
        "Needed qualities:\n" +
        "- quality 1\n" +
        "- quality 2\n" +
        "- quality 3\n" +
        "- quality 4\n" +
        "- quality 5\n" +
        "Needed skills:\n" +
        "- skill 1\n" +
        "- skill 2\n" +
        "- skill 3\n" +
        "- skill 4\n" +
        "- skill 5\n" +
        "\nReplace each placeholder with the appropriate item."
    )
    reasoning_response = openai_client.responses.create(
        model="gpt-5-nano",
        input=prompt,
    )
    print(reasoning_response.output_text)
    return reasoning_response.output_text


def embed_query(openai_client: OpenAI, query: str) -> tuple[np.ndarray, str]:
    """Generate qualities and skills for the query using ``gpt-5-nano`` and
    embed that list using OpenAI's embedding model.

    This replaces the previous behaviour where the raw user query was embedded
    directly. By asking ``gpt-5-nano`` to expand the query into concrete
    qualities and skills, the resulting embedding captures richer information
    about the desired candidate profile.
    """

    # First, derive the qualities and skills from the query.
    text_to_embed = generate_qualities_and_skills(openai_client, query)
    print("gpt-5-nano output obtained. Embedding the result...")

    # Now embed the text returned by ``gpt-5-nano``.
    embedding_response = openai_client.embeddings.create(
        input=text_to_embed,
        model=OPENAI_EMBEDDING_MODEL
    )
    embeds = np.array(embedding_response.data[0].embedding).reshape(1, -1)
    print(f"gpt-5-nano output embedded with model : {OPENAI_EMBEDDING_MODEL}")
    return embeds, text_to_embed

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

def format_name(name: str) -> str:
    """
    Format the name of the celebrity so that it is nice in the UI.
    """
    # Remove unwanted characters
    name = name.replace('\n', ' ').replace('\r', '').replace('\t', '').replace('-', ' ').strip()
    # Limit the size of the name to 50 characters
    # If longer than 50 characters, truncate it and add "..."
    if len(name) > 100:
        name = name[:97] + "..."
    return name

def get_celebs_from_indices(indices: list, celeb_to_text: dict) -> dict:
    """
    Get the resumes from the stored using a list of indices.
    /data/resume folder contains all the actual resumes.
    Each line is a JSON object representing an item.
    """
    # Parse the json lines and extract the relevant fields
    selected_celebs = {}
    for local_index, db_index in enumerate(indices):
        print("Preparing celebrity", local_index + 1, "of", len(indices))
        celeb_key = list(celeb_to_text.keys())[db_index]
        print("Celebrity : ", celeb_key)
        celeb_resume_content = celeb_to_text[celeb_key]
        celeb_name = format_name(celeb_key)
        image_path = f"{RESUME_FOLDER}/{celeb_key}.png"
        print(image_path)
        # Read the image and encode it as base64 so it can be sent directly
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        encoded_png = base64.b64encode(image_bytes).decode("utf-8")
        # Create a dictionary entry with the relevant fields
        selected_celebs[db_index] = {
            "name": celeb_name,
            "png_base64": encoded_png,
            "resume_content": celeb_resume_content,
        }
    print("Retrieving done.")
    return selected_celebs

def rerank_items(cohere_client: cohere.Client, top_n: int, query: str, indices: list, celeb_to_text: dict) -> list:
    """
    Rerank the celebs using Cohere's reranking model.
    The reranking model uses the query and the items to compute a relevance score for each item.
    The items are then sorted based on the relevance score.
    """
    selected_celebs = get_celebs_from_indices(indices, celeb_to_text)
    # Rerank the items using Cohere
    print("Reranking items...")
    response = cohere_client.rerank(
        query=query,
        documents=[selected_celebs[idx]["resume_content"] for idx in selected_celebs],
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
    celeb_to_text = load_dict(CELEB_TO_TEXT_PATH)
    openai_client = OpenAI()
    cohere_client = cohere.ClientV2(COHERE_API_KEY)
    # Store in the app's state
    app.state.text_vector_db = text_vector_db
    app.state.celeb_to_text = celeb_to_text
    app.state.openai_client = openai_client
    app.state.cohere_client = cohere_client


# Define the endpoint for searching the text defining items
@app.post("/search")
async def search_items(request: QueryRequest):
    """
    Search for items based on the user's query.
    Perform the following steps:
    1. Use ``gpt-5-nano`` to extract relevant qualities and skills and embed them.
    2. Retrieve the top k items from the FAISS Text Embedding vector database.
    3. Rerank the items using Cohere's reranking model.
    4. Get the items from the indices.
    5. Return the reasoning text along with the items to the user.
    """
    try:
        start_time = time.time()
        query = request.query
        print(f"STEP 1: Received query: {query}")
        print("==========================")
        print("==========================")
        print(f"STEP 2 - Start: Embedding query: {query}")
        embedded_query, reasoning_text = embed_query(openai_client=app.state.openai_client, query=query)
        print(f"STEP 2 - End: Query embedded in the following vector: {embedded_query}")
        print("==========================")
        print("==========================")
        print(f"STEP 3 - Start: Retrieving from FAISS index of size {app.state.text_vector_db.ntotal}")
        indices = retrieve_indices(index=app.state.text_vector_db, query_vector=embedded_query, k=NUMBER_OF_RETRIEVED_ITEMS)
        print(f"STEP 3 - End: Retrieved {NUMBER_OF_RETRIEVED_ITEMS} items")
        print("==========================")
        print("==========================")
        print(f"STEP 4 - Start: Reranking the {NUMBER_OF_RETRIEVED_ITEMS} retrieved items")
        reranked_indices = rerank_items(cohere_client=app.state.cohere_client, top_n=NUMBER_OF_OUTPUT_ITEMS, query=query, indices=indices, celeb_to_text=app.state.celeb_to_text)
        print(f"STEP 4 - End: Reranked the items and outputted the top {NUMBER_OF_OUTPUT_ITEMS} items")
        print("==========================")
        print("==========================")
        print(f"STEP 5 - Start: Extract attributes of the {NUMBER_OF_OUTPUT_ITEMS} from /data/resume folder")
        result_celebs = get_celebs_from_indices(indices=reranked_indices, celeb_to_text=app.state.celeb_to_text)
        print(f"STEP 5 - End: Extracted the attributes of the items")
        print(f"List of ordered celebrity names: {[result_celebs[idx]['name'] for idx in reranked_indices]}")
        print("==========================")
        print("==========================")
        print("TOTAL PROCESS TIME: {:.2f} seconds".format(time.time() - start_time))
        return {"reasoning": reasoning_text, "results": [result_celebs[idx] for idx in reranked_indices]}
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
