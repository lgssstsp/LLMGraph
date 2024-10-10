
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


parser = argparse.ArgumentParser("main")
parser.add_argument('--user_req', type=str, default='', help='None')
parser.add_argument('--learning_rate', type=float, default=0.02, help='None')
parser.add_argument('--epoch', type=int, default=400, help='None')
parser.add_argument('--gpu', type=int, default=0, help='None')
args = parser.parse_args()
user_req = args.user_req


user_req = """I have an graph, it is saved on the dataset file: Cora,
in which node represent the paper and edges represent the citation relationship.
The node attribute is the keywords mentioned in the paper.
I want to predict the domain of the given paper.
I think SAGEConv will be useful.
"""


gpu = args.gpu
print("="*25,"INSTRUCTION","="*25+"\n")
print(user_req)
print("\n"+"="*25,"INSTRUCTION END","="*25+"\n")


retriever = Retrieval()
knowledge_agent = KnowledgeAgent()


is_revise = True
while is_revise:
    start_time = time.time()
    planning_agent = PlanningAgent(retriever=retriever)
    planning_results = planning_agent.execute(user_req)


    knowledge_preprocess(knowledge_agent, planning_results)
    configure_retriever(knowledge_agent, retriever)   
    
    
    data_agent = DataAgent(retriever=retriever)
    feature_engineering = data_agent.execute(user_req, planning_results)


    configuration_agent = ConfigurationAgent(retriever=retriever)
    space, space_construction, algo = configuration_agent.execute(user_req, planning_results)


    search_agent = SearchingAgent(retriever=retriever)
    searched_arch_file, search_summary, path = search_agent.execute(planning_results, gpu, space, algo, feature_engineering, args)

    tune_agent = TuningAgent(retriever=retriever)
    tune_file_path, tune_summary = tune_agent.execute(planning_results, path, gpu, searched_arch_file, space, feature_engineering, args)

    end_time = time.time()
    consume_time = end_time - start_time
 
    response_agent = ResponseAgent(retriever=retriever)
    response = response_agent.execute(searched_arch_file, tune_file_path, consume_time, path)


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

    is_revise = planning_agent.revise_loop(planning_results, response)

