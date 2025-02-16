
import os
import sys
import re
import json
import ast
import time
# from langchain.llms import BaseLLM
from langchain.text_splitter import HTMLHeaderTextSplitter
import llm_api
from pydantic import BaseModel, Field
from typing import Optional
from typing import Any
from retrive import Retrieval
from utils import evaluate_case_num
import torch
# from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree, is_undirected, to_undirected
import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon, PPI, Reddit, Coauthor, CoraFull, gnn_benchmark_dataset, Flickr, CitationFull, Amazon, Actor, CoraFull
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class DataAgent(BaseModel):
    args:Any = Field(..., description="Args")
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    html_file_path: str = Field(default="PyG/pygtransforms.html", description="Path to the HTML file containing PyG documentation.")
    html_splitter: HTMLHeaderTextSplitter = None 
    content: Optional[str] = None 
    data_analysis: Optional[dict] = None 
    agent_profile: str = Field(default="""
        # Profile
        You are a Graph Learning Specialist with specialized skills in navigating and utilizing document structures for optimizing machine learning workflows. Your expertise includes extracting, parsing, and interpreting complex data from documents formats to assist in feature engineering.

        # Objective
        Your task is to analyze the PyG documentation and user requests to identify and select appropriate feature engineering techniques that are most effective for the specified task plan. This involves extracting relevant techniques from the documentation that directly align with the user's objectives and requirements, thereby enhancing the model's performance.

        # Functions
        1. **Feature Engineering Selection Function**
        - **Purpose**: To determine the best feature engineering techniques from the provided documentation that align with the user's request and task requirements.
        - **Input**: User request \('user\_req'\), task plan details \('task\_plan'\), specific documentation content \('content'\), and the statistical characteristics of data \('graph_structural_analysis'\).\\
        - **Output**: A list of up to three selected feature engineering techniques, formatted as a Python dictionary ensuring they are directly applicable and beneficial for enhancing the model's performance.

        # Human Expertise
        Human experts are crucial in the process of selecting effective feature engineering techniques from PyG documentation. Their expertise involves analyzing the task requirements, understanding the types of datasets involved, and comprehending detailed descriptions of feature engineering. This knowledge enables them to choose the most relevant and beneficial feature engineering functions that align with the specific needs of the task and enhance the overall performance of the model. These decisions are based on a deep understanding of how different techniques can affect the efficiency and effectiveness of graph neural network models in various contexts.
        """, description="The profile of this agent")

    class Config:
        arbitrary_types_allowed = True 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "Header 2")])
        self.content = self.load_content()

    def load_content(self):
        """ Load and split HTML content from file. """
        
        try:
            html_header_splits = self.html_splitter.split_text_from_file(self.html_file_path)
            return html_header_splits[2]  # Assuming the relevant content is always in the third section.
        except OSError as e:
            os.chdir('..')
            html_header_splits = self.html_splitter.split_text_from_file(self.html_file_path)
            return html_header_splits[2]
    
    def analyze_dataset(self, planning_results, save_path):
        """Analyze the dataset and return related statistics."""
        dataset_name = planning_results.get('Data', None)
        path = 'NODE_LEVEL/data/'
        data_analysis = {}
        if not dataset_name:
            return data_analysis
        
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
        elif dataset_name in ['actor']:
            dataset = Actor(path + 'Actor/')
            data = dataset[0]
        elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=dataset_name, root=path)
            data = dataset[0]
        elif dataset_name in ['Photo', 'Computer']:
            if dataset_name == 'Computer':
                dataset = Amazon(path + 'AmazonComputers', 'Computers')
                data = dataset[0]
            elif dataset_name == 'Photo':
                dataset = Amazon(path + 'AmazonPhoto', 'Photo')
                data = dataset[0]
        else:
            return data_analysis
        
        data_analysis = {}

        data_analysis["num_nodes"] = data.num_nodes
        data_analysis["num_edges"] = data.num_edges
        data_analysis["num_node_features"] = data.num_node_features
        data_analysis["num_classes"] = dataset.num_classes

        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        data_analysis["avg_node_degree"] = node_degrees.float().mean().item()
        data_analysis["max_node_degree"] = node_degrees.max().item()
        data_analysis["min_node_degree"] = node_degrees.min().item()
        

        degree_counts = torch.bincount(node_degrees.long())

        G = nx.from_edgelist(data.edge_index.t().tolist())
        clustering_coeff = nx.clustering(G)

        if nx.is_connected(G):
            data_analysis["diameter"] = nx.diameter(G)
        else:
            components = list(nx.connected_components(G))

        graph_density = data.num_edges / (data.num_nodes * (data.num_nodes - 1) / 2)
        data_analysis["graph_density"] = graph_density

        path_lengths = []
        for source, target_lengths in nx.all_pairs_shortest_path_length(G):
            for target, length in target_lengths.items():
                path_lengths.append(length)
        avg_shortest_path_length = np.mean(path_lengths)
        data_analysis["avg_shortest_path_length"] = avg_shortest_path_length


        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        data_analysis["avg_node_degree"] = node_degrees.float().mean().item()
        data_analysis["max_node_degree"] = node_degrees.max().item()
        data_analysis["min_node_degree"] = node_degrees.min().item()
        
        degree_dict = {i: node_degrees[i].item() for i in range(data.num_nodes)}
        data_analysis["node_degrees"] = degree_dict

        fig, axs = plt.subplots(3, 1, figsize=(5, 12)) 

        degree_counts = torch.bincount(node_degrees.long())
        axs[0].plot(degree_counts.tolist(), label='Node Degree Distribution')
        axs[0].set_title('Node Degree Distribution', fontsize=14)
        axs[0].set_xlabel('Degree', fontsize=12)
        axs[0].set_ylabel('Frequency', fontsize=12)
        axs[0].legend(loc='upper right')

        node_features = data.x.numpy()
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(node_features)

        axs[1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=node_degrees.numpy(), cmap='viridis')
        axs[1].set_title(f'Node Feature Visualization (PCA)', fontsize=14)
        axs[1].set_xlabel('Component 1', fontsize=12)
        axs[1].set_ylabel('Component 2', fontsize=12)
        # axs[1].colorbar(label='Node Degree')

        num_nodes_to_show = int(data.num_nodes * 0.01)
        degree_threshold = 6
        node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
        nodes_to_include = torch.nonzero(node_degrees > degree_threshold).squeeze().tolist()

        if len(nodes_to_include) > num_nodes_to_show:
            nodes_to_include = nodes_to_include[:num_nodes_to_show]

        edge_index = data.edge_index
        mask_0 = torch.isin(edge_index[0], torch.tensor(nodes_to_include))
        mask_1 = torch.isin(edge_index[1], torch.tensor(nodes_to_include))
        edge_mask = mask_0 & mask_1  
        subgraph_edges = edge_index[:, edge_mask]

        subgraph = nx.Graph()
        subgraph.add_edges_from(subgraph_edges.t().tolist())

        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(subgraph, pos, with_labels=True, node_size=50, node_color='skyblue', font_size=8, edge_color='gray', width=0.5, ax=axs[2])
        axs[2].set_title(f"Subgraph Visualization", fontsize=14)

        plt.tight_layout() 
        
        plt.show()
        plt.savefig(save_path)
        plt.close(fig)

        return data_analysis
    

        
    def execute(self, user_req, planning_results):
        revision_plan = self.args.preference
        
        select_path_prompt = """
            Given the fold list is $FOLDLIST:{fold_list}.
            Each element in this list $FOLDLIST corresponds to node-level task, graph-level task, and link-level task respectively.
            The given level of task is $TASKLEVEL:{task_level}.
            Your job is to select the suitable element in $FOLDLIST that is equivalent to $TASKLEVEL:{task_level} and the format of your output should be a dictionary as follows.
            path:Considering the task level $TASKLEVEL, output an appropriate element from the given list $FOLDLIST. 
            Output only one element from the given list $FOLDLIST, just the name of the element, and do not include any other information.
            
            """
        feature_engineering_prompt = """
            Given the user request "{user_req}" and the task plan outlined as key-value pairs in "{planning_results}", 
            as well as the revision plan $REVISION_PLAN: {revision_plan} based on previous experimental results.
            your task is to define appropriate feature engineering and data loading parameters to improve the model's performance.
            Please detail the configurations as instructed below:
            - Feature Engineering: Refer to the provided PyG documentation content "{content}" and select up to three feature engineering techniques that best align with the user's requirements. These techniques should be pertinent to the specific task described.

            Example response format:{{"feature_engineering": ["Technique1", "Technique2", "Technique3"]}}
            It is crucial that the output strictly follows the JSON format specified in the example. Ensure your response is well-structured as JSON objects, matching the example response format perfectly.
            Remember: The techniques you select must be verifiable within the PyG documentation content "{content}". Your choices should be directly relevant and demonstrably beneficial to enhancing the model's performance as per the user's objectives and the task's requirements.
            """
        data_analysis_root = './knowledge_base/datasets'
        
        max_retries = self.max_retries

        print('\n'+'='*25, "DATA AGENT", '='*25+'\n')
        
        # data_analysis = {}
        data_analysis_json_file = planning_results['Data'].replace(' ', '_').lower() + '.json'
        data_analysis_path = os.path.join(data_analysis_root, data_analysis_json_file)
        save_path = f'./figures/loop_{self.args.current_loop}_data_statistic.png'
        if os.path.exists(data_analysis_path):
            if not os.path.exists(save_path):
                data_analysis = self.analyze_dataset(planning_results, save_path)
            else:
                with open(data_analysis_path, 'r') as f:
                    data_analysis = json.load(f)
                os.system(f'cp ./figures/loop_1_data_statistic.png {save_path}')
        else:
            data_analysis = self.analyze_dataset(planning_results, save_path)
            with open(data_analysis_path, 'w') as f:
                json.dump(data_analysis, f, indent=4)
        
        print("\n\nData Analysis: ")
        self.data_analysis = data_analysis
        for drop_key in ["node_degrees"]:
            if drop_key in data_analysis:
                data_analysis.pop(drop_key)
        print("*"*25, "DTAT AGENT ANALTSIS", "*"*25)
        print(json.dumps(data_analysis, indent=4))
        print("*"*25, "DTAT AGENT ANALTSIS END", "*"*25)
        print("\n\n")

        headers_to_split_on=[("h2","Header 2")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        html_header_splits = html_splitter.split_text_from_file('PyG/pygtransforms.html')
        content = html_header_splits[2] # 1:General Transforms, 2: Graph Transforms ....
        code_path = ['./NODE_LEVEL/','./LINK_LEVEL/','./GRAPH_LEVEL/']
        for attempt in range(max_retries):
            response = llm_api.call_llm(select_path_prompt.format(fold_list=code_path,task_level=planning_results['Learning_tasks_on_graph']))
            selected_path = response['answer']
            selected_path = selected_path.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\{\s*.*?\s*\}', selected_path, re.DOTALL) 
            # match = re.search(r'\{"path": ".+?"\}', selected_path)
            if match:
                json_str = match.group()
                json_str = json_str.replace('\'', '\"')
                
                try:
                    json_data = json.loads(json_str)
                    path = json_data['path']
                    break
                except KeyError:
                    pass
            elif attempt == max_retries-1:
                print("\nOutput error. Program aborted")
                sys.exit()


        print("*"*25, "DTAT AGENT KNOWLEDGE", "*"*25)

        
        for attempt in range(max_retries):
            #####pyg_transforms
            df = self.retriever.pyg_transforms_knowledge.copy()

            transforms_prompts = []
            for index, row in df.iterrows():
                json_str = row['sentence']
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(json_str)
                    except (ValueError, SyntaxError) as e:
                        print(f"Index {index} Error：{e}")
                        continue  
                name = data.get('name', '')
                description = data.get('description', '')
                prompt = f"name: {name} and description: {description}"
                transforms_prompts.append(prompt)

            final_transforms_prompt = """Below are the names and descriptions of all feature engineering functions:\n"""+ "\n".join(transforms_prompts)
            
            print("Information from PyG document:\n", final_transforms_prompt)
            
            
            if self.args.num_cases > 0:
                if self.args.is_knowledge_types_filter == True:
                    ###retrive###
                    cases = self.retriever.search(feature_engineering_prompt.format(user_req = user_req, planning_results = planning_results, content=content, revision_plan=revision_plan), 
                                                'prior', 
                                                top_k=self.args.num_cases)
                    types_num = evaluate_case_num(cases, self.agent_profile, self.args.num_cases)
                    
                    
                    
                    print(f"\nKnowledge types and number among {self.args.num_cases}: {types_num}")

                    # print(cases)
                    final_cases = []
                    for case,score,sub_type in cases:
                        print("types_num:", types_num)
                        
                        try:
                            if types_num[sub_type] <= 0:
                                continue
                        except:
                            continue    
                        
                        types_num[sub_type] = types_num[sub_type] - 1
                        try:
                            if isinstance(case, str):
                                case_json = json.loads(case)
                            else:
                                case_json = case
                            print("\nCases retrieved from prior knowledge base:")
                            for drop_key in ["Dataset Link","Rank","External Data","Paper Link","Contact","Code Link","Date","Local Paper PDF Path"]:
                                if drop_key in case_json:
                                    case_json.pop(drop_key)
                            case_str = json.dumps(case_json, indent=4)
                            print(json.dumps(case_json, indent=4))
                            final_cases.append((case_str, score, sub_type))
                        except json.JSONDecodeError as e:
                            print("Error decoding JSON:", e)
                        # print()  
                    
                    cases_prompt = ''
                    for index, (case, distance, sub_type) in enumerate(final_cases, start=1):
                        cases_prompt = cases_prompt + f'This is the {index}-th case for your reference:\n' + case
                else:
                    ###retrive###
                    cases = self.retriever.search(feature_engineering_prompt.format(user_req = user_req, planning_results = planning_results, content=content, revision_plan=revision_plan), 
                                                'prior', 
                                                top_k=self.args.num_cases)

                    for case,score,sub_type in cases:
                        try:
                            # print(type(case))
                            if isinstance(case, str):
                                case_json = json.loads(case)
                            else:
                                case_json = case
                            print("\nCases retrieved from prior knowledge base:")
                            for drop_key in ["Dataset Link","Rank","External Data","Paper Link","Contact","Code Link","Date","Local Paper PDF Path"]:
                                if drop_key in case_json:
                                    case_json.pop(drop_key)
                            case_str = json.dumps(case_json, indent=4)
                            print(json.dumps(case_json, indent=4))
                        except json.JSONDecodeError as e:
                            print("Error decoding JSON:", e)
                    
                    cases_prompt = ''
                    for index, (case, distance, sub_type) in enumerate(cases, start=1):
                        cases_prompt = cases_prompt + f'This is the {index}-th case for your reference:\n' + case
                    
            else:
                cases_prompt = ''
            print("*"*25, "DTAT AGENT KNOWLEDGE END", "*"*25)
            
            prompt = feature_engineering_prompt.format(user_req = user_req, planning_results = planning_results, content=final_transforms_prompt, revision_plan=revision_plan) + cases_prompt
            
            
            response = llm_api.call_llm(prompt)
            feature_engineering = response['answer']
            feature_engineering = feature_engineering.replace('\n', '').replace('```', '').replace('\\', '')
            feature_engineering = re.sub(r',\s*}', '}', feature_engineering)
            feature_engineering = re.sub(r',\s*\]', ']', feature_engineering)
            # match = re.search(r'\{(.+?)\}', feature_engineering)
            match = re.search(r'\{\s*.*?\s*\}', feature_engineering, re.DOTALL)
            if match:
                try: 
                    json_str = match.group()
                    json_data = json.loads(json_str)
                    feature_engineering = json_data['feature_engineering']
                    print('\n'+'*'*25, "DTAT AGENT OUTPUT", '*'*25+'\n')
                    print('\n', "[Data Agent Output]")
                    print("The Feature Engineering Selected:\n", feature_engineering)
                    print('\n'+'*'*25, "DTAT AGENT OUTPUT END", '*'*25+'\n')
                    # print(json_data)
                    break
                except KeyError:
                    pass
            elif attempt == max_retries-1:
                print("Output error. Program aborted")
                sys.exit()
        
        print('\n'+'='*25, "DATA AGENT END", '='*25+'\n')
        return feature_engineering


    
    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return "custom_llm"



