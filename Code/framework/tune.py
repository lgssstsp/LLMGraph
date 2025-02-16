import os
import sys
import numpy as np
import re
import json
import time
import llm_api
import ast
import time
import argparse
from langchain_community.document_loaders import PythonLoader
from langchain.llms import BaseLLM
from langchain.text_splitter import HTMLHeaderTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from typing import Any
from retrive import Retrieval


class TuningAgent(BaseLLM, BaseModel):
    args:Any = Field(..., description="Args")
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    agent_profile: str = Field(default="""
		# Profile
		You are a Graph Learning Specialist skilled in automating neural architecture search (NAS) and tuning for graph neural networks. Your capabilities extend to managing entire lifecycle processes from architecture search, fine-tuning, to summarizing outcomes into actionable insights.

		# Objective
		Your primary task is to automate the search and tuning of neural architectures, streamline execution processes, and generate detailed, structured summaries of the outcomes.
		
		# Functions
		1. **Experimental Validation**
		- **Purpose**: Integrates and automates the complete process of neural architecture search and fine-tuning.
		- **Input**: User-defined parameters including search space, feature engineering functions, GPU preferences.
		- **Output**: Execution scripts, search logs, and tuning logs from both NAS and fine-tuning processes.
		
		2. **Summary Generation**
		- **Purpose**: Analyzes logs and data from both the NAS and fine-tuning processes to synthesize comprehensive summaries that incorporate prediction outcomes, architecture details, optimized hyperparameters, and resource usage.
		- **Input**: Logs and outputs generated during the NAS and fine-tuning phases.
		- **Output**: Python dictionary with detailed summaries highlighting key results, strategic insights, and areas for optimization.
		
		# Human Expertise
		Human experts play a crucial role in setting up and configuring the execution codes to align with the selected feature engineering functions and the search space. Human experts also critically analyze the results to ensure detailed and actionable summaries are generated, thus providing a deeper understanding of the effectiveness of the tested network architectures.
        """, description="The profile of this agent")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return "custom_llm"

    def execute(self, planning_results, path, gpu, searched_arch_file, space, feature_engineering_list, args):
        tune_output_prompt="""
            The file contains the outcomes of fine-tuning a graph neural network's architecture. 
            It details the refined network structure and the optimized hyperparameters achieved through the tuning process.
            Your job is to generate one paragraph to summarize the file as follows:{tuned_results}.
            """
        
        data = planning_results['Data']
        aggregation_operation_list = space['AGGREGATION']
        aggregation_operation_str = ''
        for aggregation_operation in aggregation_operation_list:
            aggregation_operation_str = aggregation_operation_str + aggregation_operation + ' '
        
        activation_operation_list = space['ACTIVATION']
        activation_operation_str = ''
        for activation_operation in activation_operation_list:
            activation_operation_str = activation_operation_str + activation_operation + ' '
        

        feature_engineering_str = ''
        for feature_engineering in feature_engineering_list:
            feature_engineering_str = feature_engineering_str + feature_engineering + ' '


        if path in ['./NODE_LEVEL/']:
            print('\n'+'='*25, "TUNING AGENT", '='*25+'\n')
            print('\n'+"*"*25, "START TUNE", "*"*25)

            command = (
                "python -u fine_tune.py --gpu {} --data {} "
                "--arch_filename {} --NA_PRIMITIVES {} --FEATURE_ENGINEERING {} --ACTIVATION {}"
            ).format(
                gpu, data, searched_arch_file, aggregation_operation_str, feature_engineering_str, activation_operation_str
            )
            # print(command)
            os.system(command)
 

            print('\n'+"*"*25, "TUNE DONE", "*"*25+'\n')
            tune_results_path = 'tuned_logs/tune_results_path.txt'
            tmp = open(tune_results_path).readlines()[-1].strip()
            tune_file = ' '
            if "Finsh tunining" in tmp:
                tune_file = tmp.strip().split(' ')[-1]
                tune_file = tune_file.replace('.txt', '_finalresults.txt')
                tune_file_content = PythonLoader(tune_file).load()[0].page_content

            response = llm_api.call_llm(tune_output_prompt.format(tuned_results=tune_file_content))
            summary = response['answer']
            print('\n'+"="*25, 'TUNE SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'TUNE SUMMARY END',"="*25+'\n')
            print('\n'+'='*25, "TUNING AGENT END", '='*25+'\n')

        
        elif path in ['./GRAPH_LEVEL/']:
            print('\n'+'='*25, "TUNE AGENT", '='*25+'\n')

            print("\nexecute code:")
            print('\n'+"*"*25, "START TUNE", "*"*25)

                
            command = (
                "python -u fine_tune.py --gpu {} --data {} --arch_filename {} --NA_PRIMITIVES {} --FEATURE_ENGINEERING {} "
                "--ACTIVATION {}"
            ).format(
                gpu, data, searched_arch_file, aggregation_operation_str, feature_engineering_str, activation_operation_str
            )
            # print(command)
            os.system(command)
            
            
            print("*"*25, "TUNE DONE", "*"*25+'\n')
            tune_results_path = 'tuned_logs/tune_results_path.txt'
            tmp = open(tune_results_path).readlines()[-1].strip()
            tune_file = ' '
            if "Finsh tunining" in tmp:
                tune_file = tmp.strip().split(' ')[-1]
                tune_file = tune_file.replace('.txt', '_finalresults.txt')
                tune_file_content = PythonLoader(tune_file).load()[0].page_content
            response = llm_api.call_llm(tune_output_prompt.format(tuned_results=tune_file_content))
            summary = response['answer']
            print('\n'+"="*25, 'TUNE SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'TUNE SUMMARY END',"="*25+'\n')
            print('\n'+'='*25, "TUNING AGENT END", '='*25+'\n')

        elif path in ['./LINK_LEVEL/']: 
            tune_file = 'None'
            summary = 'The network architectures obtained through random search are already well-tuned, eliminating the need for further fine-tuning.'

            print('\n'+"="*25, 'TUNE SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'TUNE SUMMARY END',"="*25+'\n')
        return tune_file, summary




