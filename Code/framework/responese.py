import os
import sys
import re
import llm_api
import json
import pandas as pd
from langchain.llms import BaseLLM
from langchain.text_splitter import HTMLHeaderTextSplitter
from pydantic import BaseModel, Field
from typing import Optional
from typing import Any
from retrive import Retrieval



class ResponseAgent(BaseLLM, BaseModel):
    args:Any = Field(..., description="Args")
    retriever: Any = Field(..., description="An instance of the Retrieval class")
    max_retries: int = Field(default=50, description="Maximum number of retries for calling the LLM.")
    agent_profile: str = Field(default="""
        # Profile
        You are a Graph Learning Specialist tasked with synthesizing various information from neural network tuning and architecture search into a structured format. Your capabilities include interpreting complex data outputs, generating comprehensive summaries, and structuring these into actionable insights.

        # Objective
        Your primary objective is to consolidate the results of graph neural network tuning and architecture searches into a clear, structured summary that outlines prediction outcomes, architecture details, hyperparameters, and resource consumption.

        # Functions
        1. ** Response Generation**
        - **Purpose**: To aggregate and synthesize information and generate a comprehensive response.
        - **Input**: Data from tuning results, architecture files, and other relevant agents.
        - **Output**: A Python Dictionary that clearly delineates prediction results, architecture specifics, optimized hyperparameters, and resource usage.

        # Human Expertise
        Human experts guide the process by ensuring the accurate interpretation of data, the applicability of the synthesized information, and the correctness of the output format. 
        """, description="The profile of this agent")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _generate(self, prompt, **kwargs):
        response = llm_api.call_llm(prompt)
        return response['answer']

    def _llm_type(self):
        return "custom_llm"

    def execute(self, searched_arch_file, tune_file_path, consume_time, path):
        response_prompt = """
            You have been provided with the following information:
            - Tune results: {Predict_results}.
            - The architecture of the searched graph neural network (GNN): {Searched_GNN}
            - The resource consumption during the learning process: {Resource_consuming}s
            Your task is to synthesize this information into a comprehensive summary of the graph learning procedure. 
            Your dictionary should include the following key components:
            1.predict_results: The prediction outcomes of the GNN.
            2.searched_GNN: The detailed architecture of the searched GNN.
            3.hyper_parameters: The optimized hyper-parameters identified during the tuning process.
            4.resource_consuming: Provide a detailed description of the resources used during the learning process, such as time. 
            Ensure that each key in the dictionary is accompanied by a concise and clear description of its corresponding element. 
            The expected output should be structured as below:
            {{
            "predict_results": "...",
            "searched_GNN": "...",
            "hyper_parameters": "...",
            "resource_consuming": "..."
            }}
            Please present your analysis in the form of a structured Python dictionary. 
            """
        
        print('\n'+'='*25, "RESPONSE AGENT", '='*25+'\n')

        if path in ['./NODE_LEVEL/', './GRAPH_LEVEL/']:

            with open(searched_arch_file, 'r', encoding='UTF-8') as searched_arch:
                searched_arch_str = searched_arch.read()
            searched_arch = searched_arch_str.split(",")[1]
            searched_arch = searched_arch.split("=")[1]
            

            with open(tune_file_path, 'r', encoding='UTF-8') as tune_file:
                tune_file = tune_file.read()
            reply = llm_api.call_llm(response_prompt.format(Predict_results=tune_file, Searched_GNN=searched_arch, Resource_consuming=consume_time))
            response = reply['answer']


            response = response.replace('\n', '').replace('```', '').replace('\\', '').lower()
            response = response.lower()
            match = re.search(r'\{\s*.*\s*\}', response, re.DOTALL)
            
            if match:
                response_str = match.group()
                print(response_str)
            else:
                print("\nNo dictionary found. Program aborted")
                sys.exit()


        elif path in ['./LINK_LEVEL/']:

            data = pd.read_csv(searched_arch_file)
            data = data.loc[data['rmse'].idxmin()]
            searched_arch = data.iloc[1:10]
            tune_file = data.iloc[-6:]



            reply = llm_api.call_llm(response_prompt.format(Predict_results=tune_file, Searched_GNN=searched_arch, Resource_consuming=consume_time))
            response = reply['answer']

            response = response.replace('\n', '').replace('```', '').replace('\\', '').lower()
            response = response.lower()
            match = re.search(r'\{\s*.*\s*\}', response, re.DOTALL)
            if match:
                response_str = match.group()
                print(response_str)
            else:
                print("\nNo dictionary found. Program aborted")
                sys.exit()

        print('\n'+'='*25, "RESPONSE AGENT END", '='*25+'\n')
