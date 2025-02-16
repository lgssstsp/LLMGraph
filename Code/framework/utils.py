import threading
import sys
from knowledge import KnowledgeAgent
from retrive import Retrieval
import re
import json
import numpy as np
from llm_api import call_llm

def knowledge_preprocess(knowledge_agent, planning_results):
    print('\n'+'='*25, "KNOWLEDGE AGENT", '='*25+'\n')
    # load existing knowledge from knowledge bases
    knowledge_agent.load_knowledge_base('prior')
    knowledge_agent.load_knowledge_base('experiment')
    knowledge_agent.load_pyg_info()
    
    print('\n'+'='*25, "KNOWLEDGE AGENT END", '='*25+'\n')

def configure_retriever(knowledge_agent, retriever):
    for knowledge_type in ['experiment', 'prior', 'pyg_nn', 'pyg_transforms']:
        knowledge = knowledge_agent.get_knowledge(knowledge_type)
        retriever.load_knowledge(knowledge, knowledge_type)
        
        

def evaluate_knowledge_importance(data, agent_profile):
    prompt = f"""Given the agent profile: {agent_profile},
    please evaluate the importance of each knowledge type
    from the following data and provide a score between 0 and 1 for each sub_type.
    Please provide your response in a JSON format where each key is the sub_type and the value is the score.\n"""
   
    for sentence, _, sub_type in data:
        prompt += f"Sub_type: {sub_type}, Sentence: {sentence}\n"
    prompt += "[End of data]"


    max_retries = 20
    for attempt in range(max_retries):
        response = call_llm(prompt)['answer']
        response = response.replace('\n', '').replace('```', '').replace('\\', '')

        match = re.search(r'\{\s*.*?\s*\}', response, re.DOTALL)
        if match:         
            try: 
                json_str = match.group()
                json_data = json.loads(json_str)
                sub_type_scores = {k: float(v) for k, v in json_data.items()}
                    
                total_score = sum(sub_type_scores.values())
                normalized_scores = {sub_type: score / total_score for sub_type, score in sub_type_scores.items()}

                return normalized_scores
            except:
                pass


def evaluate_knowledge_importance(cases, agent_profile, total_num):
    
    prompt = f"""Given the agent profile: {agent_profile},
    please evaluate the importance of each knowledge type
    from the following data and provide a score between 0 and 1 for each sub_type.
    Please provide your response in a JSON format where each key is the sub_type and the value is the score.\n"""
   
    for sentence, _, sub_type in cases:
        prompt += f"Sub_type: {sub_type}, Sentence: {sentence}\n"
    prompt += "[End of data]"

    max_retries = 20
    for attempt in range(max_retries):
        response = call_llm(prompt)['answer']
        response = response.replace('\n', '').replace('```', '').replace('\\', '')

        match = re.search(r'\{\s*.*?\s*\}', response, re.DOTALL)
        if match:         
            try: 
                json_str = match.group()
                json_data = json.loads(json_str)
                sub_type_scores = {k: float(v) for k, v in json_data.items()}
                    
                total_score = sum(sub_type_scores.values())
                normalized_scores = {sub_type: score / total_score for sub_type, score in sub_type_scores.items()}

                return normalized_scores
            except:
                pass

def evaluate_case_num(cases, agent_profile, total_num):
    prompt = f"""Given the agent profile: {agent_profile},
    and a total number of {total_num} cases, please allocate the number of cases each knowledge type should have
    from the following data. 
    Provide a number for each sub_type that reflects the proportion of the total cases they should receive, 
    which represents the importance of this type of knowledge. 
    The sum of all numbers must equal {total_num} and should be non-negative integers.
    Return your response in a JSON format 
    where each key is the sub_type and the value is the number of cases.
    You must carefully check that your output is a dictionary and that its keys are only Sub_type.
    \n"""
    for sentence, _, sub_type in cases:
        prompt += f"Sub_type: {sub_type}, Sentence: {sentence}\n"
    prompt += "[End of data]"
    response = call_llm(prompt)  
    response_text = response['answer']  
    match = re.search(r'\{\s*.*?\s*\}', response_text, re.DOTALL)
    if match:
        try:
            json_str = match.group()
            allocations = json.loads(json_str)
            sub_type_allocations = {k: int(v) for k, v in allocations.items()}

            if sum(sub_type_allocations.values()) == total_num:
                return sub_type_allocations
            else:
                print("The sum of all allocations does not match the total number of cases.")
        except json.JSONDecodeError:
            print("Failed to decode the JSON response.")
        except ValueError:
            print("Allocation values must be integers.")
    else:
        print("No valid JSON response was found.")
    return None        
