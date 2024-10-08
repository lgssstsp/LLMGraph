
## Installation
To install the necessary dependencies, run:

```bash
conda env create -f environment.yml
```

## Directory Structure

```plaintext
framework/
│
├── F2GNN/                # AUTOML source code directory
├── LRGNN/                # AUTOML source code directory
├── F2GNN/                # AUTOML source code directory
│ 
├── all-MiniLM-L6-v2/     # Word embedding model
│ 
├── knowledge_base/       # Knowledge base file directory
│   └── experiment_knowledge   # Experimental knowledge
│   └── prior_knowledge        # Prior knowledge
│ 
├── task.py              # Planning agent
├── data.py              # Data agent
├── configuration.py     # Configuration agent
├── search.py            # Searching agent
├── tune.py              # Tuning agent
├── response.py          # Response agent
├── knowledge.py         # Knowledge agent
│
├── main.py              # Entry file
│
├── llm_api.py           # Call LLM
└── retrive.py           # Retriever
```

## Usage

To run the project, navigate to the `framework` directory and execute:

```bash
cd framework
python main.py
```





