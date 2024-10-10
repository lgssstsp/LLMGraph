import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import faiss  
import pickle
import warnings
import json

warnings.filterwarnings("ignore")

class Retrieval:
    def __init__(self, model_name='all-MiniLM-L6-v2', storage_path='embeddings_store'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.experiment_knowledge = pd.DataFrame(columns=['index', 'sentence', 'embedding'])
        self.prior_knowledge = pd.DataFrame(columns=['index', 'sentence', 'embedding'])
        self.pyg_nn_knowledge = pd.DataFrame(columns=['index', 'sentence', 'embedding'])
        self.pyg_transforms_knowledge = pd.DataFrame(columns=['index', 'sentence', 'embedding'])
        
        self.experiment_knowledge_index = None  
        self.prior_knowledge_index = None  
        self.pyg_nn_knowledge_index = None
        self.pyg_transforms_knowledge_index = None
        
        self.storage_path = storage_path


    def load_knowledge(self, knowledge_dict, knowledge_type):
        if knowledge_type not in ['experiment', 'prior', 'pyg_nn', 'pyg_transforms']:
            raise ValueError("Keyword error")
        
        # print(knowledge_dict)
        for key, value in knowledge_dict.items():
            sentence = json.dumps(value, indent=4)  
            self.add_sentence(sentence, knowledge_type)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_sentence(self, sentence):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(embedding, p=2, dim=1).cpu().numpy()

    def add_sentence(self, sentence, knowledge_type):
        embedding = self.encode_sentence(sentence)
        
        experiment_knowledge_index_length = len(self.experiment_knowledge)
        prior_knowledge_index_length = len(self.prior_knowledge)
        pyg_nn_knowledge_index_length = len(self.pyg_nn_knowledge)
        pyg_transforms_knowledge_index_length = len(self.pyg_transforms_knowledge)

        if knowledge_type == 'experiment':
            new_row = pd.DataFrame([[experiment_knowledge_index_length, sentence, embedding.flatten()]], columns=['index', 'sentence', 'embedding'])
            self.experiment_knowledge = pd.concat([self.experiment_knowledge, new_row], ignore_index=True)
        elif knowledge_type == 'prior':
            new_row = pd.DataFrame([[prior_knowledge_index_length, sentence, embedding.flatten()]], columns=['index', 'sentence', 'embedding'])
            self.prior_knowledge = pd.concat([self.prior_knowledge, new_row], ignore_index=True)
        elif knowledge_type == 'pyg_nn':
            new_row = pd.DataFrame([[pyg_nn_knowledge_index_length, sentence, embedding.flatten()]], columns=['index', 'sentence', 'embedding'])
            self.pyg_nn_knowledge = pd.concat([self.pyg_nn_knowledge, new_row], ignore_index=True)
        elif knowledge_type == 'pyg_transforms':
            new_row = pd.DataFrame([[pyg_transforms_knowledge_index_length, sentence, embedding.flatten()]], columns=['index', 'sentence', 'embedding'])
            self.pyg_transforms_knowledge = pd.concat([self.pyg_transforms_knowledge, new_row], ignore_index=True)
            
        self._update_index()

        
    def _update_index(self):    
        experiment_knowledge_embeddings = self.experiment_knowledge['embedding'].values.tolist()
        prior_knowledge_embeddings = self.prior_knowledge['embedding'].values.tolist()
        pyg_nn_knowledge_embeddings = self.pyg_nn_knowledge['embedding'].values.tolist()
        pyg_transforms_knowledge_embeddings = self.pyg_transforms_knowledge['embedding'].values.tolist()
        
        
        if experiment_knowledge_embeddings:
            experiment_knowledge_embeddings = np.vstack(experiment_knowledge_embeddings)
            if self.experiment_knowledge_index is None:
                dimension = experiment_knowledge_embeddings.shape[1]
                self.experiment_knowledge_index = faiss.IndexFlatL2(dimension)  
                
            self.experiment_knowledge_index.reset()
            self.experiment_knowledge_index.add(experiment_knowledge_embeddings)
        
        if prior_knowledge_embeddings:
            prior_knowledge_embeddings = np.vstack(prior_knowledge_embeddings)
            if self.prior_knowledge_index is None:
                dimension = prior_knowledge_embeddings.shape[1]
                self.prior_knowledge_index = faiss.IndexFlatL2(dimension)  
                
            self.prior_knowledge_index.reset()
            self.prior_knowledge_index.add(prior_knowledge_embeddings)
        
        if pyg_nn_knowledge_embeddings:
            pyg_nn_knowledge_embeddings = np.vstack(pyg_nn_knowledge_embeddings)
            if self.pyg_nn_knowledge_index is None:
                dimension = pyg_nn_knowledge_embeddings.shape[1]
                self.pyg_nn_knowledge_index = faiss.IndexFlatL2(dimension)
            self.pyg_nn_knowledge_index.reset()
            self.pyg_nn_knowledge_index.add(pyg_nn_knowledge_embeddings)
        
        if pyg_transforms_knowledge_embeddings:
            pyg_transforms_knowledge_embeddings = np.vstack(pyg_transforms_knowledge_embeddings)
            if self.pyg_transforms_knowledge_index is None:
                dimension = pyg_transforms_knowledge_embeddings.shape[1]
                self.pyg_transforms_knowledge_index = faiss.IndexFlatL2(dimension)
            self.pyg_transforms_knowledge_index.reset()
            self.pyg_transforms_knowledge_index.add(pyg_transforms_knowledge_embeddings)

        

    def _save_embeddings_to_storage(self):
        os.makedirs(self.storage_path, exist_ok=True)
        filepath = os.path.join(self.storage_path, 'embeddings.pkl')
        
        stored_data = {
            'experiment': self.experiment_knowledge.to_dict(orient='records'),
            'prior': self.prior_knowledge.to_dict(orient='records'),
            'pyg_nn': self.pyg_nn_knowledge.to_dict(orient='records'),
            'pyg_transforms': self.pyg_transforms_knowledge.to_dict(orient='records')
        }

        with open(filepath, 'wb') as f:
            pickle.dump(stored_data, f)

    def search(self, query_sentence, knowledge_type, top_k=5):

        if knowledge_type == 'experiment':
            index = self.experiment_knowledge_index
        elif knowledge_type == 'prior':
            index = self.prior_knowledge_index
        elif knowledge_type == 'pyg_nn':
            index = self.pyg_nn_knowledge_index
        elif knowledge_type == 'pyg_transforms':
            index = self.pyg_transforms_knowledge_index
        else:
            raise ValueError("Invalid knowledge type")
        
        query_embedding = self.encode_sentence(query_sentence)

        distances, indices = index.search(query_embedding, top_k)
        
        if knowledge_type == 'experiment':
            knowledge_df = self.experiment_knowledge
        elif knowledge_type == 'prior':
            knowledge_df = self.prior_knowledge
        elif knowledge_type == 'pyg_nn':
            knowledge_df = self.pyg_nn_knowledge
        elif knowledge_type == 'pyg_transforms':
            knowledge_df = self.pyg_transforms_knowledge
            
            
        
        results = [(knowledge_df.iloc[idx]['sentence'], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

    def load_cases_from_directory(self, directory_path):
        sentences_to_add = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.case'):
                    case_file_path = os.path.join(root, file)
                    with open(case_file_path, 'r', encoding='utf-8') as f:
                        sentence = f.read().strip()
                        if sentence and sentence not in self.experiment_knowledge['sentence'].values and \
                           sentence not in self.prior_knowledge['sentence'].values:
                            sentences_to_add.append(sentence)

        if sentences_to_add:
            for sentence in sentences_to_add:
                self.add_sentence(sentence, 'experiment')  

