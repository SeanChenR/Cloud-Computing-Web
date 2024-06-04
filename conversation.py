# TWCC ffm-Llama model 
# api key and api end point on twcc official web 
# use FFM-Llama-70B-chat-V2 model from TWCC

import json
import requests
import configparser

# enter api key and api endpoint from TWCC website
config = configparser.ConfigParser()
config.read('config.ini')

MODEL_NAME = config['llama3']['MODEL_NAME']
API_KEY = config['llama3']['API_KEY']
API_URL = config['llama3']['API_URL']
API_HOST = config['llama3']['API_HOST']

# parameters
max_new_tokens = 500
temperature = 0.01
top_k = 10
top_p = 1
frequence_penalty = 1.03

# FFM-Llama-chat conversation api
def conversation(system, contents):
    headers = {
        "content-type": "application/json", 
        "X-API-KEY": API_KEY,
        "X-API-HOST": API_HOST}
    
    roles = ["human", "assistant"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for index, content in enumerate(contents):
        messages.append({"role": roles[index % 2], "content": content})
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "frequence_penalty": frequence_penalty
        }
    }

    result = ""
    try:
        response = requests.post(API_URL + "/models/conversation", json=data, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text, strict=False)['generated_text']
        else:
            print("error")
    except:
        print("error")
    return result.strip("\n")
