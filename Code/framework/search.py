
import os
import sys
import pandas as pd
import argparse
import re
import json
import time
import llm_api
import ast
import time
import shutil
from langchain_community.document_loaders import PythonLoader
from langchain.llms import BaseLLM
from langchain.text_splitter import HTMLHeaderTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from typing import Any
from retrive import Retrieval
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['MKL_THREADING_LAYER'] = 'GNU'


class SearchingAgent(BaseLLM, BaseModel):
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    agent_profile: str = Field(default="""
        # Profile
        You are a Graph Learning Specialist with advanced capabilities in automating neural architecture search (NAS) for graph neural networks. Your skills include configuring execution code, performing neural architecture searching operation, and generating insightful summaries from the search processes.

        # Objective
        Your primary task is to streamline the process of efficiently executing search procedures to discover optimal network architectures, and effectively summarize the outcomes using the details derived from search logs.

        # Functions
        1.**Argument Configuration**
        - **Purpose**: Sets up the parameters and arguments that control the search process.
        - **Input**: User-defined parameters including search space, feature engineering functions, GPU preferences.
        - **Output**: A configured execution script ready for the NAS (Neural Architecture Search)  process.

        2.**Code Execution** 
        - **Purpose**: Employs advanced AutoML techniques to systematically explore network architectures.
        - **Input**: The configured execution script.
        - **Output: Search log from the exploration of network architectures.

        3. **Summary Generation** 
        - **Purpose**: Analyzes logs from the NAS (Neural Architecture Search) process to construct detailed summaries.
        - **Input**: Logs and data generated during the NAS process.
        - **Output**: Summaries that capture key results and strategic insights, aiding in the evaluation of the search outcomes.

        # Human Expertise
        Human experts are essential in setting up and configuring the execution codes tailored to the chosen feature engineering functions and the constructed search space. Following the completion of the AutoML process, they critically assess and interpret the log data to create comprehensive summaries that encapsulate the search findings and provide valuable insights into the effectiveness of the tested architectures.
        """, description="The profile of this agent")


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return None


    def search_p12(self, planning_results, data, gpu, space, algo, feature_engineering_list, args):
        print("feature_engineering_list: ",feature_engineering_list)
        
        max_retries = self.max_retries
        select_path_prompt = """
            Given the fold list is $FOLDLIST:{fold_list}.
            Each element in this list $FOLDLIST corresponds to the code path of node-level task, graph-level task, and link-level task respectively.
            The given level of task is $TASKLEVEL:{task_level}.
            Your job is to select the proper element in $FOLDLIST that is equivalent to $TASKLEVEL:{task_level} and the format of your output should be a dictionary as follows.
            path:Considering the task level $TASKLEVEL, output an appropriate element from the given list $FOLDLIST. Output only one element from the given list $FOLDLIST.
            For example, if the task level is 'link-level', you must choose the fold './ProfCF/' and output {{'path':'./ProfCF/'}}, if the task level is 'graph-level', you must choose the fold './LRGNN/' and output {{'path':'./LRGNN/'}}, if the task level is 'node-level', you must choose the fold './F2GNN/' and output {{'path':'./F2GNN/'}}.
            """
    

        cur_time = time.strftime("%m%d_%H%M%S", time.localtime())
        
        code_path = ['./F2GNN/','./LRGNN/','./ProfCF/']

        for attempt in range(max_retries):
            response = llm_api.call_llm(select_path_prompt.format(fold_list=code_path,task_level=planning_results['Learning_tasks_on_graph']))
            selected_path = response['answer']
            selected_path = selected_path.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\{\s*.*?\s*\}', selected_path, re.DOTALL) 
            if match:
                json_str = match.group().replace('\'',"\"")
                
                json_data = json.loads(json_str)
                try:
                    path = json_data['path']
                    # print("\nThe Path LLM Selected:", json_data)
                    break
                except KeyError:
                    pass

            elif attempt == max_retries-1:
                print("\nOutput error. Program aborted")
                sys.exit()
        
        
        os.chdir(path)

        aggregation_operation_list = space['AGGREGATION']
        aggregation_operation_str = ''
        for aggregation_operation in aggregation_operation_list:
            aggregation_operation_str = aggregation_operation_str + aggregation_operation + ' '
        feature_engineering_str = ''
        
        
        for feature_engineering in feature_engineering_list:
            print("feature_engineering: ",feature_engineering)
            feature_engineering_str = feature_engineering_str + feature_engineering + ' '



        if path in ['./F2GNN/']:
            print('\n'+"*"*25, "START SEARCH", "*"*25)
            

            command = (
                "python -u train_search.py --gpu {} --data {} --NA_PRIMITIVES {} --FEATURE_ENGINEERING {} "
                "--search_agg True --learning_rate {} --epochs {} | tee search_logs/search_{}_{}.txt | tee print.txt"
            ).format(
                gpu, data, aggregation_operation_str, feature_engineering_str, 
                args.learning_rate, args.epoch, cur_time, data
            )
            os.system(command)
            
            output_file_name = 'search_logs/search_{}_{}.txt'.format(cur_time,data)
        
        elif path in ['./LRGNN/']:
            readout_list = space['READOUT_GRAPH']
            readout_str = ''
            for readout in readout_list:
                readout_str = readout_str + readout + ' '
            print('\n'+"*"*25, "START SEARCH", "*"*25)


            command = (
                "python -u train_search.py --gpu {} --data {} --NA_PRIMITIVES {} --FEATURE_ENGINEERING {} --READOUT_PRIMITIVES {}"
                "--search_agg True --learning_rate {} --epochs {} | tee search_logs/search_{}_{}.txt | tee print.txt"
            ).format(
                gpu, data, aggregation_operation_str, feature_engineering_str, readout_str, 
                args.learning_rate, args.epoch, cur_time, data
            )
            os.system(command)

            output_file_name = 'search_logs/search_{}_{}.txt'.format(cur_time,data)

        elif path in ['./ProfCF/']:
            aggregation_operation_list = space['AGGREGATION']
            
            config_file_path = "run/grids/design/cf.txt"
            params = {
                'dataset.name d': [[data]],
                'dataset.encoder_dim i':[[64]],
                'gnn.msg m': [['identity']],
                'gnn.layer_type g': [aggregation_operation_list],         
                'gnn.layers_mp l': [[4]],
                'gnn.stage_type s': [['concat']],
                'model.edge_decoding in': [['concat']],
                'gnn.act a': [space['ACTIVATION']],
                'gnn.component_num cn': [[4]],
                'gnn.component_aggr ca': [['concat']]
            }

            with open(config_file_path, 'w') as f:
                for key, values_list in params.items():
                    formatted_values = '[' + ','.join(
                        (str(value[0]) if isinstance(value[0], int) else f"'{value[0]}'") if len(value) == 1 
                        else ','.join(str(v) if isinstance(v, int) else f"'{v}'" for v in value)
                        for value in values_list
                    ) + ']'
                    f.write(f'{key} {formatted_values}\n')

            print('\n'+"*"*25, "START RANDOM SEARCH", "*"*25)
            os.chdir("run")
 
            os.system("bash run_batch.sh --feature_engineering_list \"{}\"".format(feature_engineering_str))
                
                
            file_path = 'results/cf_grid_cf/agg/test.csv'
            data = pd.read_csv(file_path)
            best_row = data.loc[data['rmse'].idxmin()]
            param_mapping = {
                'dataset': 'd',
                'init_dim': 'i',
                'msg': 'm',
                'gnn_layer': 'g',
                'layers_num': 'l',
                'stage': 's',
                'inter_func': 'in',  
                'act': 'a',
                'cpnt_num': 'cn',
                'cpnt_aggr': 'ca'
            }

            config_parts = []
            for key, abbreviation in param_mapping.items():
                if key in best_row.index:
                    value = best_row[key]
                    config_parts.append(f"{abbreviation}={value}")

            config_str = "cf-" + "-".join(config_parts)
            print("\nConfig string based on best structure:", config_str)
            output_file_name = config_str

        print("*"*25, "SEARCH DONE", "*"*25+'\n')
        return output_file_name, path

    def search_p34(self, output_file_name, planning_results, path):

        differentiable_search_output_prompt="""
            Your job is to generate one paragraph to summarize the results.
            1. We have executed the code of neural architecture search (NAS).
            2. The description of data and user preferences are in the task plan {task_plan}.
            3. The searched architectures are {searched_arch}.
            4. The training logging is {search_log}.
            """

        random_search_output_prompt="""
            Your job is to generate one paragraph to summarize the results.
            1. We have executed the code of neural architecture search (NAS).
            2. The description of data and user preferences are in the task plan:{task_plan}.
            3. The best searched architectures are {searched_arch}.
            4. The training logging of the best GNN architecture on the stp of random search is: {search_log}.
            """

        if path in ['./F2GNN/']:
            tmp = open(output_file_name).readlines()[-1].strip()
            searched_arch = ' '
            if "searched res for" in tmp:
                searched_arch_file = tmp.strip().split(' ')[-1]
                searched_arch = PythonLoader(searched_arch_file).load()[0].page_content
            arch = searched_arch.split(',')[1].split('=')[-1]
            with open(output_file_name, 'r') as file:
                lines = file.readlines()
            total_lines = len(lines)
            max_lines = 20
            if total_lines > max_lines:
                first_lines = lines[:int(max_lines/2)] if total_lines >= int(max_lines/2) else lines
                last_lines = lines[-int(max_lines/2):] if total_lines >= int(max_lines/2) else lines
                search_logging = ''.join(first_lines + last_lines)
            else:
                search_logging = ''.join(lines)
            
            response = llm_api.call_llm(differentiable_search_output_prompt.format(task_plan=planning_results,
                                                                    searched_arch = searched_arch,
                                                                    search_log = search_logging))
            summary = response['answer']
            print('\n'+"="*25, 'SEARCH SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'SEARCH SUMMARY END',"="*25+'\n')

        if path in ['./LRGNN/']:
            tmp = open(output_file_name).readlines()[-1].strip()
            searched_arch = ' '
            if "searched res for" in tmp:
                searched_arch_file = tmp.strip().split(' ')[-1]
                searched_arch = PythonLoader(searched_arch_file).load()[0].page_content
            arch = searched_arch.split(',')[1].split('=')[-1]
            
            with open(output_file_name, 'r') as file:
                lines = file.readlines()
            total_lines = len(lines)
            max_lines = 20
            if total_lines > max_lines:
                first_lines = lines[:int(max_lines/2)] if total_lines >= int(max_lines/2) else lines
                last_lines = lines[-int(max_lines/2):] if total_lines >= int(max_lines/2) else lines
                search_logging = ''.join(first_lines + last_lines)
            else:
                search_logging = ''.join(lines)

            response = llm_api.call_llm(differentiable_search_output_prompt.format(task_plan=planning_results,
                                                                    searched_arch = searched_arch,
                                                                    search_log = search_logging))
            summary = response['answer']
            print('\n'+"="*25, 'SEARCH SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'SEARCH SUMMARY END',"="*25+'\n')

        elif path in ['./ProfCF/']:
            search_log_path = f'results/cf_grid_cf/{output_file_name}/agg/train/stats.json'
            all_logs = {}
            try:
                with open(search_log_path, 'r') as file:
                    for i, line in enumerate(file):
                        try:
                            all_logs[str(i)] = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"\nError: {e}")
                total_lines = len(all_logs)
                search_logging = []
                if total_lines > 10:
                    for i in range(total_lines):
                        if i < 5 or i >= total_lines - 5:
                            search_logging.append(all_logs[str(i)])
                else:
                    for i in range(total_lines):
                        search_logging.append(all_logs[str(i)])
            except FileNotFoundError:
                print(f"\nFile not found: {search_log_path}")
            except Exception as e:
                print(f"\nError when open file: {e}")


            file_path = 'results/cf_grid_cf/agg/test.csv'
            data = pd.read_csv(file_path)
            searched_arch = data.loc[data['rmse'].idxmin()]
            config = searched_arch
            arch = {
                'Message_Function': config.msg,
                'Aggregation': config.gnn_layer,
                'Layer_Combination': config.stage,
                'Interaction_Function': config.inter_func,
                'Activation': config.act,
                'Component_Combination': config.cpnt_aggr,
            }


            response = llm_api.call_llm(random_search_output_prompt.format(task_plan=planning_results,
                                                                            searched_arch = searched_arch,
                                                                            search_log = search_logging))
            summary = response['answer']
            print('\n'+"="*25, 'SEARCH SUMMARY',"="*25+'\n')
            print(summary)
            print('\n'+"="*25, 'SEARCH SUMMARY END',"="*25+'\n')
            searched_arch_file = file_path
        return searched_arch_file, summary

    def execute(self, planning_results, gpu, space, algo, feature_engineering_list, args):
        print('\n'+'='*25, "SEARCHING AGENT", '='*25+'\n')
        data = planning_results['Data']
        # print("data: ",data)
        output_file_name, path = self.search_p12(planning_results, data, gpu, space, algo, feature_engineering_list, args)
        searched_arch_file, summary = self.search_p34(output_file_name, planning_results, path)
        print('\n'+'='*25, "SEARCHING AGENT END", '='*25+'\n')
        return searched_arch_file, summary, path


