"""Wrapper Embedding model APIs."""
import json
import requests
from typing import List
from pydantic.v1 import BaseModel
from langchain_core.embeddings.embeddings import Embeddings

class CustomEmbeddingModel(BaseModel, Embeddings):
    base_url: str = "http://localhost:12345"
    api_key: str = ""
    model: str = ""
    def get_embeddings(self, payload):
        endpoint_url = f"{self.base_url}/models/embeddings"
        headers = {
            "Content-type": "application/json",
            "accept": "application/json",
            "X-API-KEY": self.api_key,
            "X-API-HOST": "afs-inference"
        }
        response = requests.post(endpoint_url, headers=headers, data=payload)
        body = response.json()
        datas = body["data"]
        embeddings = [data["embedding"] for data in datas]

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({
            "model": self.model,
            "inputs": texts
        })
        emb = self.get_embeddings(payload)
        emb_list = [[0 if x is None else x for x in sublist] for sublist in emb]
        return emb_list


    def embed_query(self, text: str) -> List[List[float]]:
        payload = json.dumps({
            "model": self.model,
            "inputs": [text]
        })
        emb = self.get_embeddings(payload)[0]
        emb_list = [0 if x is None else x for x in emb]
        return emb_list

def get_embeddings_model(API_URL, API_KEY, MODEL_NAME):
    embeddings_model = CustomEmbeddingModel(
        base_url = API_URL,
        api_key = API_KEY,
        model = MODEL_NAME,
    )
    return embeddings_model

if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')

    API_KEY = config['embedding']['API_KEY']
    API_URL = config['embedding']['API_URL']
    MODEL_NAME = config['embedding']['MODEL_NAME']
    embeddings_model = get_embeddings_model(API_URL, API_KEY, MODEL_NAME)
    query = "早安你好"
    embeddings = embeddings_model.embed_query(query)
    print(embeddings)