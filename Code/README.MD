# Environment
    environment.yml

# Directory Structure
    framework/ 
    ├── NODE_LEVEL/ # automl source code 
    ├── GRAPH_LEVEL/ # automl source code 
    ├── LINK_LEVEL/ # automl source code 
    ├── all-MiniLM-L6-v2/ # Pretrained language model, can be downloaded from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 
    ├── knowledge_base/ # Knowledge base file directory 
    ├── manager.py # Manager agent 
    ├── data.py # Data agent 
    ├── configuration.py # Configuration agent 
    ├── search.py # Evaluation agent 
    ├── tune.py # Evaluation agent 
    ├── response.py # Evaluation agent 
    ├── knowledge.py # Knowledge agent 
    ├── main.py # Entry point file 
    ├── llm_api.py # LLM API call 
    └── retrive.py # Retriever

# Run Example
    cd framework
    python main.py --dataset actor