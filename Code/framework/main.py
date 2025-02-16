
import os
import sys
import time
import json
import ast
from task import PlanningAgent
from data import DataAgent
from configuration import ConfigurationAgent
from search import SearchingAgent
from tune import TuningAgent
from responese import ResponseAgent
from retrive import Retrieval
from knowledge import KnowledgeAgent
from utils import knowledge_preprocess, configure_retriever
import argparse
import subprocess
import threading



# command
parser = argparse.ArgumentParser("main")
parser.add_argument('--dataset', type=str, default='', help='None')
parser.add_argument('--user_req', type=str, default='', help='None')
parser.add_argument('--learning_rate', type=float, default=0.02, help='None')
parser.add_argument('--epoch', type=int, default=400, help='None')
parser.add_argument('--gpu', type=int, default=4, help='None')
parser.add_argument('--num_cases', type=int, default=3, help='None')
parser.add_argument('--knowledge_percentage', type=float, default=1.0, help='None')
parser.add_argument('--knowledge_types', nargs='+', default=['ogb','pyg_info_nn','new_paper'], help='None')
# parser.add_argument('--knowledge_types', nargs='+', default=['new_paper'], help='None')
parser.add_argument('--is_knowledge_types_filter', action='store_false', help='Enable the knowledge types filter')

parser.add_argument('--preference', type=str, default='None', help='None')
parser.add_argument('--current_loop', type=int, default=1, help='None')
args = parser.parse_args()
user_req = args.user_req



dateset_dict = {
'actor':"""I have graph, it is saved on the dataset file: actor, 
which is an actor-only induced subgraph of the film-director-actor-writer network. 
In this graph-based dataset, nodes represent actors, and edges represent 
the co-occurrence of two actors on the same Wikipedia page. 
Node features correspond to keywords extracted from the actors' Wikipedia pages. 
The goal is to use graph neural networks to classify nodes into one of five categories 
based on the words present in an actor's Wikipedia page.""",

'ogbg-molhiv':"""I have a graph, it is saved on the dataset file: ogbg-molhiv,
where each graph represents a molecular structure. 
In this dataset, nodes correspond to atoms 
and edges represent chemical bonds between these atoms. 
The goal is to predict whether a molecule inhibits HIV replication, 
an essential task for antiretroviral drug discovery. 
Each graph is labeled with a binary indicator (0 or 1), 
where 1 indicates that the molecule has anti-HIV activity 
in bioassays. """,

'epinions':"""I have a graph, it is saved on the dataset file: epinions, 
in which nodes represent the users and the edges represent 
the trust relationship between users. 
The nodeattribute could include user activity metrics, 
ratings, or other relevant informationthat signifies 
the user's influence or trustworthiness. 
I want to predict the potential of a trust 
relationship forming between two users.""",

'amazon-sports':"""I have a dataset named amazon-sports, 
which is a subset of the sports category 
in an e-commerce dataset. 
In this graph-based dataset, nodes represent products or users, 
and edges represent the purchasing behaviors between 
users or the associations between products. 
I aim to use graph neural networks for link prediction 
to predict which products might be purchased 
together or which users might be interested in certain products.""",

'NCI1':"""I have a dataset, it is saved on the dataset file: NCI1. 
In the NCI1 dataset, each graph represents a chemical compound, 
where nodes correspond to atoms within the compound, 
and edges represent the chemical bonds between atoms. 
I want to find one GNN that has better accuracy 
for predicting the classification or molecular properties of a compound.
I think SAGEConv will be useful.""",


'ogbn-arxiv':"""I have a graph, it is saved on the dataset file: ogbn-arxiv, 
in which each node represents an arXiv paper and edges represent the citation relationship between papers.
I want to predict the research domain of each paper using a graph neural network.""",

'cora':"""I have an graph, it is saved on the dataset file: Cora,
in which node represent the paper and edges represent the citation relationship.
The node attribute is the keywords mentioned in the paper.
I want to predict the domain of the given paper.
I think SAGEConv will be useful.""",

'proteins':"""I have a graph, it is saved on the dataset file: PROTEINS. 
This dataset contains protein features and structures, 
I want to find one GNN that has better accuracy 
for predicting the classification of these proteins
""",

'DD':"""I have a graph, it is saved on the dataset file: DD. 
This dataset contains protein features and structures, 
I want to find one GNN that has better accuracy 
for predicting the classification of these proteins
""",


'computer':"""I have a graph-based dataset saved on the dataset file: Computer, 
which is a subset of the Amazon co-purchase graph. 
In this dataset, nodes represent goods, 
and edges indicate that two goods are frequently bought together. 
Node features are bag-of-words encoded product reviews, 
and class labels are given by the product category. 
I aim to use graph neural networks for product categorization 
to predict the category of products based on the co-purchase 
pattern and product reviews.""",

 'photo':""" I have a dataset saved on the dataset file: Photo. 
 This dataset exclusively involves photographic equipment and supplies. 
 Nodes represent photo-related products, 
 and edges show frequent co-purchases among them. 
 Node features may include bag-of-words encoded product reviews, 
 and class labels are based on the product category. 
 The objective is to apply graph neural networks for 
 classifying these products into appropriate categories, 
 guided by their co-purchase behavior and review information.""",

'genius':"""I have graph, it is saved on the dataset file: genius, 
which is an actor-only induced subgraph of the film-director-actor-writer network. 
In this graph-based dataset, nodes represent actors, and edges represent 
the co-occurrence of two actors on the same Wikipedia page. 
Node features correspond to keywords extracted from the actors' Wikipedia pages. 
The goal is to use graph neural networks to classify nodes into one of five categories 
based on the words present in an actor's Wikipedia page.""",


}



user_req = dateset_dict[args.dataset]



gpu = args.gpu

print("="*25,"ARGS","="*25+"\n")
print("knowledge_types: ", args.knowledge_types)
print("knowledge_percentage: ", args.knowledge_percentage)
print("\n"+"="*25,"ARGS END","="*25+"\n")






retriever = Retrieval()
knowledge_agent = KnowledgeAgent(args=args)


#revise loop?
is_revise = True
revise_count = 1
max_revise = 7
memory = {}
revision_plan = 'None'

while is_revise and revise_count < max_revise:
    print("&"*25,f"CURRENT LOOP {revise_count}","&"*25)
    
    print("="*25,"INSTRUCTION","="*25+"\n")
    print(user_req)
    print("\n"+"="*25,"INSTRUCTION END","="*25+"\n")
    
    args.preference = revision_plan
    args.current_loop = revise_count
    
    start_time = time.time()
    
    planning_agent_start_time = time.time()
    planning_agent = PlanningAgent(retriever=retriever, args=args)   
    
    planning_results = planning_agent.execute(user_req)
    planning_agent_end_time = time.time()
    print("planning_agent cost time:",planning_agent_end_time- planning_agent_start_time)
    # planning_results = {'Learning_tasks_on_graph': 'link-level', 'Learning_task_types': 'regression', 'Evaluation_metric': 'MSE or RMSE', 'Preference': 'None', 'Data': 'epinions'}
    
    knowledge_agent_start_time = time.time()
    # Process knowledge_agent information
    knowledge_preprocess(knowledge_agent, planning_results)
    knowledge_agent_end_time = time.time()
    print("knowledge_agent cost time:",knowledge_agent_end_time- knowledge_agent_start_time)

    
    # Configure the retriever
    configure_retriever(knowledge_agent, retriever)   
    
    
    data_agent_start_time = time.time()
    data_agent = DataAgent(retriever=retriever, args=args)
    feature_engineering = data_agent.execute(user_req, planning_results)
    planning_results['graph_structural_analysis'] = data_agent.data_analysis
    print(planning_results)
    data_agent_end_time = time.time()
    print("data_agent cost time:",data_agent_end_time- data_agent_start_time)

    # #example
    # feature_engineering = ['AddSelfLoops']
    # #\\example

    configuration_agent_start_time = time.time()
    configuration_agent = ConfigurationAgent(retriever=retriever, args=args)
    space, space_construction, algo = configuration_agent.execute(user_req, planning_results)
    configuration_agent_end_time = time.time()
    print("configuration_agent cost time:",configuration_agent_end_time- configuration_agent_start_time)
    # #example
    # space = {'AGGREGATION': ['GCNConv', 'GATConv'], 'ACTIVATION': ['relu', 'elu'], 'FUSION': ['sum', 'mean']}
    # algo = {'algorithm': 'Differentiable Search'}
    # #\\example

    experiment_agent_start_time = time.time()
    search_agent = SearchingAgent(retriever=retriever, args=args)
    searched_arch_file, search_summary, path = search_agent.execute(planning_results, gpu, space, algo, feature_engineering, args)


    tune_agent = TuningAgent(retriever=retriever, args=args)
    tune_file_path, tune_summary = tune_agent.execute(planning_results, path, gpu, searched_arch_file, space, feature_engineering, args)
    experiment_agent_end_time = time.time()
    print("experiment_agent cost time:",experiment_agent_end_time- experiment_agent_start_time)

    end_time = time.time()
    consume_time = end_time - start_time
 
    response_agent = ResponseAgent(retriever=retriever, args=args)
    response = response_agent.execute(searched_arch_file, tune_file_path, consume_time, path)

    # Update experiment results to the experiment knowledge base
    all_results = {
        "user_requery": user_req,
        "planning_results": planning_results,
        "feature_engineering": feature_engineering,
        "configuration_results": {
            "space": space,
            "algo": algo
        },
        "path": path,
        "search_results": {
            "summary": search_summary,
        },
        "tuning_results": {
            "summary": tune_summary
        },
        "response": response,
        "consume_time": consume_time
    }
    retriever.load_knowledge(all_results, 'experiment')
    knowledge_agent.save_pipeline_result(all_results)
    memory[revise_count] = all_results


    # Final step, determine whether to perform the revise loop
    is_revise, revision_plan = planning_agent.revise_loop(planning_results, response, memory)
    # print("is_revise:",is_revise)
    # print("revision_plan:",revision_plan)
    
    revise_count += 1
    print("Current revise: ",revise_count)
    print("&"*25,f" CURRENT LOOP END {revise_count} ","&"*25)
    

