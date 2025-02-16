
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import re
import json
import time
import difflib
from langchain.llms import BaseLLM
import llm_api
from pydantic import BaseModel, Field
from langchain.llms import BaseLLM
from typing import Any
from retrive import Retrieval

TOTAL_TOKEN=None


class PlanningAgent(BaseModel):
    args:Any = Field(..., description="Args")
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    prompt_template: str = Field(default="""
        As a researcher specialized in graph-structured data, you are tasked with interpreting user requests related to graph learning. Your objective is to analyze the user's request, provided here: {user_request}. 
        Your analysis should deconstruct the request into several key components and present your findings in the form of a Python dictionary. Specifically, identify and articulate the following elements:

        1. Learning_tasks_on_graph: The learning task on graph can only be node-level, link-level, or graph-level.
        Node-level: If the task aims to predict or classify attributes or behaviors of individual nodes based on their features or their relationships to other nodes, it is a node-level task. For instance, identifying the category of a paper in a citation network based on its content and citation links.
        Link-level: If the task focuses on the relationships between nodes, such as predicting the existence, strength, or quality of these connections, classify it as a link-level task. An example is predicting the rating a user gives to a movie, where the prediction is about the edge (rating) connecting two types of nodes (users and movies), which is a regression problem.
        Graph-level: If the task involves understanding or predicting properties that describe entire graphs or significant subgraphs, it is a graph-level task. Examples include classifying whole graphs into categories or predicting a property of a chemical compound represented as a graph.
        Review the user request carefully and determine whether the task is node-level, link-level, or graph-level. Provide your analysis in a structured Python dictionary format, categorizing the task correctly based on the provided details and emphasizing the focus of the learning task.

        2. Learning_task_types: Ascertain whether the task is a regression or classification problem, or any other type of learning task. Clarify the nature of the learning challenge.

        3. Evaluation_metric: Identify the evaluation metric that is desired or implied in the user's request for gauging the success of the learning task.
        If the task is a classification problem, such as predicting categories or labels or domains, suggest metrics like Accuracy, Precision, Recall, or F1-Score.
        If the task is a regression problem, such as predicting numerical values, recommend metrics like Mean Squared Error (MSE), or Root Mean Squared Error (RMSE).
        In cases where the user has not specified a preference for any metric, provide a rationale for your recommended metric based on the task's details and the desired outcomes. 

        4. Preference: Note any specific preferred operations that the user mentions. 
        If no preferences are stated, acknowledge this with 'None'.

        5. Data: Only the dataset name.
        Ensure that each component is distinctly identified and described, providing a clear and actionable breakdown of the user's request, and the format of your outputs should be a dictionary.

        You should provide a Python dictionary as the output with a flat structure (no nested dictionaries), containing only the above keys. 
        """, description="Template used for the prompt to LLM.")
    
    
    agent_profile: str = Field(default="""
        # Profile
        You are a Graph Learning Specialist with exceptional abilities in interpreting complex user instructions and translating them into structured, actionable data formats. Furthermore, once the evaluation agent has completed the assessment, the manager agent will determine whether to restart the experiments, depending on the outcomes of the evaluation agent and the experimental results from the knowledge base.\\

        # Objective
        Your task is to deconstruct user instructions related to graph learning into detailed, executable plans, ensuring all elements of the request are accurately captured and clearly categorized.

        # Functions
        1. **Task Plan Function**
        - **Purpose**: To meticulously parse and interpret user instructions, identifying key information pertinent to graph learning tasks.
        - **Input**: User instructions.
        - **Output**: A Python Dictionary of the task highlighting its level (node, link, or graph), type of learning task (classification, regression, etc.), the name of dataset, and evaluation metrics.

		2. **Revision Plan Function** 
		- **Input**: The complete output of the current experiment, previous results for comparison and the experimental results from the knowledge base. 
		- **Output**: A Python Dictionary of the revision plan, which contains the reason for the revision, direction of revision and the expected performance. 

        
        
        # Human Expertise
        Human expertise in this context involves a deep understanding of how to interpret complex user requests related to graph learning. The expertise includes:
        1. **Task Categorization**: Identifying the level of graph learning tasks (node, link, or graph-level) based on the user's description.
        2. **Task Type Determination**: Distinguishing between different types of learning tasks, such as classification or regression, based on the details provided in the user's request.
        3. **Evaluation Metric Selection**: Selecting appropriate evaluation metrics.
        4. **Preference Identification**: Noting any specific operational preferences mentioned by the user.
        5. **Dataset Identification**: Accurately identifying the relevant datasets mentioned in the user's request.
        """, description="The profile of this agent")
    

    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return "custom_llm"

    def execute(self, user_request):
        print('\n'+'='*25, "PLANNING AGENT", '='*25+'\n')
        prompt = self.prompt_template.format(user_request=user_request)
        for attempt in range(self.max_retries):
            
            

            
            planning_results = self._generate(prompt)
   
            planning_results = planning_results.replace('\n', '').replace('```', '').replace('\\', '')
            
            match = re.search(r'\{\s*.*?\s*\}', planning_results, re.DOTALL)
            if match:
                            
                try: 
                    json_str = match.group()
                    json_data = json.loads(json_str)
                    expected_keys = {
                        'Learning_tasks_on_graph': None,
                        'Learning_task_types': None,
                        'Evaluation_metric': None,
                        'Preference': None,
                        'Data': None
                    }
                    def find_closest_key(key, expected_keys):
                        closest_match = difflib.get_close_matches(key, expected_keys.keys(), n=1, cutoff=0.6) 
                        return closest_match[0] if closest_match else None

                    normalized_output = {}
                    for key, value in json_data.items():
                        # print("find_closest_key")
                        matched_key = find_closest_key(key, expected_keys)
                        if matched_key:
                            normalized_output[matched_key] = value


                    print(normalized_output)

                    print('\n'+'='*25, "PLANNING AGENT END", '='*25+'\n')
                    return normalized_output
                except:
                    # print("dont match")
                    pass
            elif attempt == self.max_retries - 1:
                print("No dictionary found. Program aborted")
                sys.exit()
    

    def revise_loop(self, planning_results, response, memory):

        ###RETRIVE###
        similar_info = self.retriever.search(str(planning_results), 'experiment', top_k=self.args.num_cases)

        experiment_case = ""
        for info in similar_info:
            experiment_case += f"```Experiment Case: {info}```\n"  #
            print(f"Experiment Case:\n{info}\n")

        prompt = f"""
            Based on the following planning results and response, please determine if a revision loop is needed:
            Planning Results: {planning_results} 
            [This represents the planning details of the current experiment defined by the planning agent before the experiment takes place.]

            Similar Information: {experiment_case} 
            [These are relevant experiment cases retrieved from the knowledge base. They provide context or examples from similar experiments that can inform the current experiment.]

            Response: {response} 
            [This is the summary which includes an assessment of the experimental process and results. ]

            Memory: {memory}
            [This is a record of previous experimental information, including the outcomes and any revision loops triggered in the past. 
            The memory helps track the experiment's history and adjustments made throughout the process.]

            Ignore any test set results of `Similar Information`/`Response`/`Memory`, focus only on the validation set, 
            as you can't access test set labels 
            during the experiment—these represent real, unseen data. 
            Evaluation and adjustments should rely solely on the validation set metrics.

            Should a revision loop be triggered? If yes, generate a revision plan. If no, return `None`.
            You should respond with a Python dictionary with the following keys:
            - 'answer': 'yes' or 'no' 
            - 'reason': the reason explaining your decision in detail, do not include nested dictionaries. 
            - 'revision_plan': either a detailed revision plan if 'yes', or `None` if 'no', do not include nested dictionaries.
            - 'expected_performance': the expected performance of the model after the revision loop, do not include nested dictionaries.
            """  
           
        
        for attempt in range(self.max_retries):
            llm_response = self._generate(prompt)
            # llm_response = llm_response.lower().strip()
            
            llm_response = llm_response.replace('\n', '').replace('```', '').replace('\\', '')
            match = re.search(r'\{\s*.*?\s*\}', llm_response, re.DOTALL)
            if match:
                            
                try: 
                    json_str = match.group()
                    json_data = json.loads(json_str)
                    
                    answer = json_data['answer']
                    reason = json_data['reason']
                    revision_plan = json_data['revision_plan']
                    expected_performance = json_data['expected_performance']
                    
                    if "yes" in json_data['answer']:
                        print('\n'+'='*25, "REVISION LOOP", '='*25)
                        print(json.dumps(json_data, indent=4))
                        print("*** Revision loop needed ***")
                        print('\n'+'='*25, "REVISION LOOP END", '='*25+'\n')
                        return True, revision_plan
                    else:
                        print('\n'+'='*25, "REVISION LOOP", '='*25)
                        print(json.dumps(json_data, indent=4))
                        print("*** No revision loop needed ***")
                        print('\n'+'='*25, "REVISION LOOP END", '='*25+'\n')
                        return False, 'None'
                
                except:
                    pass
            
            
            elif attempt == self.max_retries - 1:
                print("No dictionary found. Program aborted")
                sys.exit()
            