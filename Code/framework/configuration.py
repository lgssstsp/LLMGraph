import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import re
import json
import time
import llm_api
import ast
from langchain.text_splitter import HTMLHeaderTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from typing import Any
from retrive import Retrieval

class ConfigurationAgent(BaseModel):
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    html_file_path: str = Field(default="PyG/pygnn.html", description="Path to the HTML file containing PyG documentation.")
    agent_profile: str = Field(default="""
        # Profile
        You are a Graph Learning Specialist specialized in navigating and utilizing configuration options. Your expertise allows you to effectively parse documentation, extract operational data, and apply this information to configure search space and search algorithm.

        # Objective
        Your primary objective is to orchestrate the configuration process of graph neural networks by selecting appropriate modules, preparing operation candidates, evaluating these candidates, and finally selecting an optimal configuration that enhances the model’s effectiveness and efficiency.

        # Functions
        1. **Module Selection Function**: Aimed at identifying the best modules for inclusion in the graph neural network based on the task's specifics.
        - **Input**: Task requirements and available module options.
        - **Output**: List of modules deemed most suitable for the task.

        2. **Operation Preparation Function**: Prepares the detailed list of operations from specific documentation content (`content`) that can be performed by the selected modules.
        - **Input**: Selected modules and specific documentation content.
        - **Output**: Detailed operations capable of being executed by these modules.

        3. **Candidate Selection Function**: Evaluates the prepared operations and selects the most promising candidates for final deployment.
        - **Input**: List of prepared operations.
        - **Output**: Shortlist of candidate operations for search space.

        4. **Construct Search Space Function**: Constructs a comprehensive search space where different configurations can be tested and evaluated.
        - **Input**: Candidate operations.
        - **Output**: A structured search space.

        5. **Algorithm Selection Function**: Selects the most suitable algorithm through based on the identified requirements.
        - **Input**: The constructed search space, the task plan and the selected modules.
        - **Output**: The most effective search algorithm for finding the network architecture.

        # Human Expertise
        The configuration of graph neural networks involves a sequential process. Initially, human experts select modules that align with the specific demands of the task. Following this, they outline potential operations for these modules and narrow down the choices to the most effective ones for candidate selection. The next step is constructing a search space based on the selected modules and selected candidates. Finally, human experts select an algorithm that best navigates this space to find the optimal architecture of the network. Throughout this process, human expertise ensures that each step is tailored to meet the task-specific goals and technical requirements efficiently.  
        """, description="The profile of this agent")


    def execute(self, user_req, planning_results):
        space, space_construction = self.configuration_space_agent(planning_results)
        algo = self.configuration_algo_agent(planning_results, space_construction, space)
        return space, space_construction, algo

    
    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return "custom_llm"
    
    def configuration_space_agent(self, task_plan):
        
        micro_dict = {
            "AGGREGATION": ['GCNConv', 'GATConv'],  
            "ACTIVATION": ['relu', 'elu'], 
            "FUSION":['sum','mean'],
            "READOUT_GRAPH": ['global_sum', 'global_mean']
        }
        
        applicable_keys_prompt = """
        You are given a key "{key}" which concludes with a postfix "{postfix}". 
        This postfix signifies the task level for which the key is specifically designed. 
        Your task is to determine whether this key is suitable for a different task level "{task_level}".
        For example, postfix "GRAPH" means the key is suitable for graph-level task, postfix "NODE" means the key is suitable for node-level task, postfix "LINK" means the key is suitable for link-level task.
        Please provide your answer as a simple 'yes' or 'no' without additional explanations or text.
        """
        module_pyg_match_prompt = """
        Given the document content "{content}", your task is to identify and extract all operations mentioned. These operations are described throughout the provided content and are key components of the procedures or methods discussed in the text.
        Please list down all identified operations in a structured Python list format. Each operation should be a separate string element within the list, ensuring that the output is clean, precise, and directly usable in a Python environment.
        Format your output as follows:["operation1", "operation2", "operation3", ...]
        Ensure that the operations are extracted accurately from the given content and the output strictly adheres to the specified Python list format.
        """
        op_select_prompt = """
        You are provided with a set of operations $OPERATIONS_SET: {op_sets}. Additionally, the user's preferences for specific operations are outlined in their task plan $TASK_PLAN: {task_plan}.
        Your task is to identify and select the appropriate operations from {op_sets} that align with the user's preferences specified in the task plan {task_plan}. 
        If the task plan's preferences are not explicitly defined or only a few operations are mentioned, you should limit your selection to those explicitly stated preferences without making additional recommendations.
        If the task plan's preferences are None, you must return an empty Python list to indicate that there are no specific operation preferences given.
        Ensure the selected operations are accurately extracted from the provided set and that your output strictly adheres to the Python list format, focusing solely on user-specified preferences.
        """

        max_retries = self.max_retries
        print('\n'+'='*25, "CONFIGURATION AGENT", '='*25)

        applicable_keys = []
        for key in micro_dict.keys():
            if "_" in key:
                postfix = key.split("_")[-1]
                llm_prompt = applicable_keys_prompt.format(task_level=task_plan['Learning_tasks_on_graph'], key=key, postfix=postfix)
                response = llm_api.call_llm(llm_prompt)
                if 'yes' in response['answer'].lower():
                    applicable_keys.append(key)
            else:
                applicable_keys.append(key)

        headers_to_split_on=[("h2","Header 2")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        html_header_splits = html_splitter.split_text_from_file('PyG/pygnn.html')
        content = html_header_splits[1] # 1:conv layer, 2: aggr func ....
        
        for attempt in range(max_retries):   
            response = llm_api.call_llm(module_pyg_match_prompt.format(content=content))
            op_sets = response['answer']    
            op_sets = op_sets.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\[\s*.*?\s*\]', op_sets, re.DOTALL)
            if match:
                op_sets_str = match.group()
                op_sets = ast.literal_eval(op_sets_str)
                break
            elif attempt == max_retries-1:
                print("\nOutput error. Program aborted")
                sys.exit()

        for attempt in range(max_retries):  
            df = self.retriever.pyg_nn_knowledge.copy()

            nn_prompts = []
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
                nn_prompts.append(prompt)

            final_nn_prompt = """Below are the names and descriptions of all Aggregation functions, which play an important role in the message passing framework of Graph Neural Networks:\n"""+"\n".join(nn_prompts)
            
            print("Information from PyG document:\n", final_nn_prompt)
            
            ###retrive###
            cases = self.retriever.search(op_select_prompt.format(op_sets=op_sets, task_plan=task_plan, dict=micro_dict), 
                                          'prior', 
                                          top_k=3)
            
            for case,score in cases:
                try:
                    if isinstance(case, str):
                        case_json = json.loads(case)
                    else:
                        case_json = case
                    case_json = json.loads(case)
                    print("\nCases retrieved from prior knowledge base:")
                    for drop_key in ["Dataset Link","Rank","External Data","Paper Link","Contact","Code Link","Date","Local Paper PDF Path"]:
                        if drop_key in case_json:
                            case_json.pop(drop_key)
                    print(json.dumps(case_json, indent=4))
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)

            
            cases_prompt = ''
            for index, (case, distance) in enumerate(cases, start=1):
                cases_prompt = cases_prompt + f'This is the {index}-th case for your reference:\n' + case
            prompt = op_select_prompt.format(op_sets=final_nn_prompt, task_plan=task_plan, dict=micro_dict) + cases_prompt
            
            
            response = llm_api.call_llm(prompt)            
            selected_aggregation_op = response['answer']
            selected_aggregation_op = selected_aggregation_op.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\[\s*.*?\s*\]', selected_aggregation_op, re.DOTALL)
            if match:
                selected_aggregation_op_str = match.group()
                selected_aggregation_op = ast.literal_eval(selected_aggregation_op_str)
                break
            else:
                selected_aggregation_op = []

        selected_op = {'AGGREGATION':selected_aggregation_op}

        space = {}
        for key_origin in micro_dict.keys():
            for key_selected in selected_op.keys():
                if key_origin == key_selected:
                    space[key_origin] = micro_dict[key_origin] + selected_op[key_selected]
                elif key_origin in applicable_keys:
                    space[key_origin] = micro_dict[key_origin]
        
        print('\n', "[Configuration Agent Output]")
        print("Constructed Search Space:", space)
        return space, applicable_keys
    


    def configuration_algo_agent(self, task, module, space):
        configuration_algorithm_prompt = """
            Based on the given task plan $TASK:{task}, design dimension $MODULE:{module}, and specific design space $SPACE:{space}, you are required to select an appropriate search algorithm. 
            The selection should be between a 'Random Search' algorithm and a 'Differentiable Search' algorithm, contingent upon the nature of the learning task within the graph context ($TASK:{task}).
            Here are the guidelines for your selection:
            - If the learning task in $TASK:{task} is either 'node-level' or 'graph-level', you are recommended to opt for the 'Differentiable Search' algorithm due to its suitability and implementation availability for these levels.
            - For 'link-level' tasks within $TASK:{task}, given the absence of 'Differentiable Search' algorithm implementations for such tasks, you should choose the 'Random Search' algorithm.
            Your output should be a Python dictionary specifying the chosen algorithm, your output should be:{{"algorithm": "the algorithm you selected"}}
            Adhere to the task details and constraints to ensure your choice aligns with the provided guidelines and is suitable for the task's specific needs.
            """
        max_retries = self.max_retries
        
        for attempt in range(max_retries):
            response = llm_api.call_llm(configuration_algorithm_prompt.format(task=task, module=module, space=space))
            search_algorithm = response['answer']
            search_algorithm = search_algorithm.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\{\s*.*?\s*\}', search_algorithm, re.DOTALL)
            if match:
                search_algorithm_str = match.group()
                search_algorithm = ast.literal_eval(search_algorithm_str)
                break
            elif attempt == max_retries-1:
                print("\nOutput error. Program aborted")
                sys.exit()
        print("Selected Search Algorithm:", search_algorithm)
        print('\n'+'='*25, "CONFIGURATION AGENT END", '='*25+'\n')
        return search_algorithm

