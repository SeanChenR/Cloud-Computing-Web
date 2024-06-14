"""Wrapper Embedding model APIs."""
import json
import requests
import configparser
from typing import List
from pydantic import BaseModel
from langchain_core.embeddings.embeddings import Embeddings

# Read config file Because the information are in config.ini
config = configparser.ConfigParser()
config.read("config.ini")

MODEL_NAME = "ffm-embedding"
API_KEY = "a4d9b574-ee0e-4369-a310-b6a053c06c67"
API_URL = "https://api-ams.twcc.ai/api/"

# TWCC embedding model
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
        # Sometimes the "post" will be disconnected
        # so if the data is not converted successfully, "post" it again.
        while True:
            try:
                response = requests.post(endpoint_url, headers=headers, data=payload)
                body = response.json()
                datas = body["data"]
                embeddings = [data["embedding"] for data in datas]
                break
            except:
                print(body)
                continue

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({
            "model": self.model,
            "inputs": texts
        })
        return self.get_embeddings(payload)


    def embed_query(self, text: str) -> List[List[float]]:
        payload = json.dumps({
            "model": self.model,
            "inputs": [text]
        })
        emb = self.get_embeddings(payload)
        return emb[0]

# the function to get embedding
def get_embeddings_model(API_URL, API_KEY, MODEL_NAME):
    embeddings_model = CustomEmbeddingModel(
        base_url = API_URL,
        api_key = API_KEY,
        model = MODEL_NAME,
    )
    return embeddings_model

# if you want to test embedding connection, run the code and you can see the result of connection
if __name__ == "__main__":
    MODEL_NAME = MODEL_NAME
    API_KEY = API_KEY
    API_URL = API_URL
    embeddings_model = get_embeddings_model(API_URL, API_KEY, MODEL_NAME)
    # print(embeddings_model.embed_query("請問台灣最高的山是？"))
    print(embeddings_model.embed_documents(["test1", "test2", "test3"]))
