import requests
import socket
import json




# # gpt api
def call_llm(prompt):
    url = ""
    payload = json.dumps({
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    
    headers = {
        'Accept': "application/json",
        'Authorization': "",  #API KEY
        'User-Agent': '',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload, timeout=300)
    ansewr = response.json()["choices"][0]["message"]["content"]
    return {"answer": ansewr}

if __name__ == "__main__":
    response = call_llm('hello?')
    print(response)

