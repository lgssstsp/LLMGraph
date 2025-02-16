import requests
import socket
import json


import requests
import json
import tiktoken

def count_tokens(prompt, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(prompt))

def split_prompt(prompt, chunk_size=2000, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(prompt)
    
    # 分割 tokens
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def call_llm(prompt):
    url = "https://www.dmxapi.com/v1/chat/completions"

    model = "gpt-4o"

    headers = {
        'Accept': "application/json",
        'Authorization': "YOUR_API_KEY",  
        
        'User-Agent': 'mtuopenai/1.0.0 (https://www.dmxapi.com)',
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    
    response = requests.post(url, headers=headers, data=payload, timeout=300)
    response_data = response.json()
    
    answer = response_data["choices"][0]["message"]["content"]

    return {"answer": answer}


if __name__ == "__main__":
    response = call_llm('hello?')
    print(response)


