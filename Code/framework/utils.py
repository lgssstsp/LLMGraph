import threading
import sys
from knowledge import KnowledgeAgent
from retrive import Retrieval

def knowledge_preprocess(knowledge_agent, planning_results):
    print('\n'+'='*25, "KNOWLEDGE AGENT", '='*25+'\n')
    # knowledge_agent.load_knowledge_base('prior')
    # knowledge_agent.load_knowledge_base('experiment')
    # knowledge_agent.load_pyg_info()
    
    
    # # # The knowledge_agent collects relevant knowledge based on the information from planning_results
    # knowledge_agent.process_ogb_leaderboard(top_k=1, task_plan=planning_results)
    # knowledge_agent.search_with_task_plan(planning_results)
    knowledge_agent.process_pyg_info('nn')
    knowledge_agent.process_pyg_info('transforms')
    print('\n'+'='*25, "KNOWLEDGE AGENT END", '='*25+'\n')

def configure_retriever(knowledge_agent, retriever):
    for knowledge_type in ['experiment', 'prior', 'pyg_nn', 'pyg_transforms']:
        knowledge = knowledge_agent.get_knowledge(knowledge_type)
        retriever.load_knowledge(knowledge, knowledge_type)