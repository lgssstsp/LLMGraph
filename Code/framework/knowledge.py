import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import PyPDF2
import pdfplumber
import llm_api
import time
import re
import random



class KnowledgeAgent:
    def __init__(self, args, base_dir='knowledge_base'):
        self.ogb_urls = [
            "https://ogb.stanford.edu/docs/leader_nodeprop",
            "https://ogb.stanford.edu/docs/leader_graphprop",
            "https://ogb.stanford.edu/docs/leader_linkprop"
        ]
        self.args = args
        
        self.pyg_url = 'https://pytorch-geometric.readthedocs.io/en/latest/'
        
        # google search engine api key
        self.api_key = "your_api_key"
        self.cse_id = "your_cse_id"


        self.base_dir = base_dir     
        self.experiment_knowledge_base = os.path.join(self.base_dir, 'experiment_knowledge')
        self.prior_knowledge_base = os.path.join(self.base_dir, 'prior_knowledge')
        
        self.background_dir = os.path.join(self.base_dir, 'background')
        self.pyg_info_dir = os.path.join(self.base_dir, 'pyg_info')

        self.experiment_knowledge_cache = {}
        self.prior_knowledge_cache = {}
        self.background_info = {}
        self.pyg_info = {}

        
        os.makedirs(self.experiment_knowledge_base, exist_ok=True)
        os.makedirs(self.prior_knowledge_base, exist_ok=True)
  

           
    def load_pyg_info(self):
        """
        Load information from JSON files stored in subdirectories of the base directory.
        Returns:
            dict: A dictionary containing information organized by subdirectory and filename.
        """
        for subdir in os.listdir(self.pyg_info_dir):
            subdir_path = os.path.join(self.pyg_info_dir, subdir)
            if os.path.isdir(subdir_path):  
                self.pyg_info[subdir] = {}
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(subdir_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            self.pyg_info[subdir][filename] = json.load(file)
             
        

    def summarize_background(self):
        background_dir = os.path.join(self.base_dir, 'background')
        pdf_files = [f for f in os.listdir(background_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_dict = {'NODE_LEVEL.pdf': 'node-level', 'LINK_LEVEL.pdf': 'graph-level', 'GRAPH_LEVEL.pdf': 'link-level'}
            pdf_path = os.path.join(background_dir, pdf_file)

            text = self.extract_text_from_pdf(pdf_path)
            
            prompt = f"""
            Please summarize the main content of the article based on the text below.
            Our project background involves configuring the search space and search algorithms of AutoML.
            This article {pdf_file}, is applicable to {pdf_dict[pdf_file]} tasks.
            Focus specifically on the methods discussed in the article, particularly the model design aspects. 
            Please exclude the part of experiments or related work or evaluation.            

            Here is the text content of the article:\n
            """ + text
            
            summary = llm_api.call_llm(prompt)

            json_file_path = os.path.join(background_dir, pdf_file.replace('.pdf', '.json'))
            with open(json_file_path, 'w') as json_file:
                json.dump({
                    'description': f'This is the background section of the experiment, applicable to {pdf_dict[pdf_file]} tasks',
                    'summary': summary
                }, json_file, ensure_ascii=False, indent=4)
        

    def load_knowledge_base(self, knowledge_type):
        """Load knowledge base files into cache based on the knowledge type"""
        if knowledge_type == 'prior':

            total_loaded = 0
            for k_type in self.args.knowledge_types:
                self.prior_knowledge_cache[k_type] = {}
                dir_path = os.path.join(self.prior_knowledge_base, k_type)
                all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]
                num_to_load = int(len(all_files) * self.args.knowledge_percentage)
                selected_files = random.sample(all_files, num_to_load)

                for filepath in selected_files:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        file_name = os.path.basename(filepath)
                        self.prior_knowledge_cache[k_type][file_name] = data  
                total_loaded += len(selected_files)
            total_files = sum(len(files) for files in self.prior_knowledge_cache.values())
            print(f"[Loaded existing prior knowledge] Loaded {total_files} files into cache overall.")


        elif knowledge_type == 'experiment':
            for root, dirs, files in os.walk(self.experiment_knowledge_base):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.experiment_knowledge_cache[file] = data
            print(f"[Loaded existing experiment knowledge] Loaded {len(self.experiment_knowledge_cache)} existing experiment knowledge base files into cache.")

        else:
            raise ValueError("Invalid type.")


    def get_webpage_content(self, url):
        """Fetch webpage content"""
        response = requests.get(url)
        response.encoding = 'utf-8'
        return response.text

    def parse_ogb_leaderboard(self, html_content):
        """Parse OGB leaderboard webpage and extract data"""
        soup = BeautifulSoup(html_content, 'html.parser')
        task_description = soup.find('h1').get_text(strip=True)
        results = []
        dataset_headers = soup.find_all('h3')

        for header in dataset_headers:
            dataset_name = header.get_text(strip=True).replace('Leaderboard for', '').strip()
            dataset_link = header.find('a', href=True)['href'] if header.find('a', href=True) else ''
            next_table = header.find_next('table')
            if next_table:
                rows = next_table.find_all('tr')[1:]  # 跳过表头
                for row in rows:
                    cols = row.find_all('td')
                    if cols:
                        data = {
                            'Task Description': task_description,
                            'Dataset Name': dataset_name,
                            'Dataset Link': dataset_link,
                            'Rank': cols[0].get_text(strip=True),
                            'Method': cols[1].get_text(strip=True),
                            'External Data': cols[2].get_text(strip=True),
                            'Test Accuracy': cols[3].get_text(strip=True),
                            'Validation Accuracy': cols[4].get_text(strip=True),
                            'Contact': cols[5].find('a')['href'] if cols[5].find('a') else '',
                            'Paper Link': cols[6].find_all('a')[0]['href'] if len(cols[6].find_all('a')) > 0 else '',
                            'Code Link': cols[6].find_all('a')[1]['href'] if len(cols[6].find_all('a')) > 1 else '',
                            'Parameters': cols[7].get_text(strip=True),
                            'Hardware': cols[8].get_text(strip=True),
                            'Date': cols[9].get_text(strip=True)
                        }
                        results.append(data)
        return results
    
    def get_knowledge(self, knowledge_type):
        """Retrieve knowledge cache based on type"""
        if knowledge_type == 'experiment':
            return self.experiment_knowledge_cache
        elif knowledge_type == 'prior':
            return self.prior_knowledge_cache
        elif knowledge_type == 'background':
            return self.background_info
        elif knowledge_type == 'pyg_nn':
            return self.pyg_info['nn']
        elif knowledge_type == 'pyg_transforms':
            return self.pyg_info['transforms']
        else:
            raise ValueError("Invalid knowledge type provided.")

    def update_knowledge(self, knowledge_type, new_data):
        """Update knowledge cache"""
        if knowledge_type == 'experiment':
            self.experiment_knowledge_cache.update(new_data)
        elif knowledge_type == 'prior':
            self.prior_knowledge_cache.update(new_data)
        else:
            raise ValueError("Invalid knowledge type provided.")
        # print(f"Updated {knowledge_type} knowledge cache with {len(new_data)} items.")

    def select_top_k_methods(self, leaderboard_data, k=5):
        """Select the top-K methods for each task and dataset"""
        df = pd.DataFrame(leaderboard_data)
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce').fillna(9999)
        grouped_df = df.groupby(['Task Description', 'Dataset Name'])
        top_k_methods = grouped_df.apply(lambda x: x.nsmallest(k, 'Rank')).reset_index(drop=True)
        return top_k_methods.to_dict(orient='records')


    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  
                    text += page_text + "\n"
        return text
        
    def save_to_knowledge_base(self, entry, knowledge_type= 'prior'):
        """Save data to the specified knowledge base and update the corresponding cache"""
        if knowledge_type == 'experiment':
            base_dir = self.experiment_knowledge_base
            cache = self.experiment_knowledge_cache
        elif knowledge_type == 'prior':
            base_dir = self.prior_knowledge_base
            cache = self.prior_knowledge_cache
        else:
            raise ValueError("knowledge_type is not valid.")

        if 'Task Description' in entry and 'Dataset Name' in entry and 'Method' in entry:
            task_description = entry['Task Description'].replace(' ', '_')
            dataset_name = entry['Dataset Name'].replace(' ', '_')
            method = entry['Method'].replace(' ', '_')
            filename = f"{task_description}_{dataset_name}_{method}.json"
        elif 'Query' in entry:
            query = entry['Query'].replace(' ', '_')
            filename = f"search_{query}.json"
        else:
            filename = "entry.json"

        if knowledge_type == 'experiment':
            filepath = os.path.join(base_dir, filename)
        elif knowledge_type == 'prior':
            filepath = os.path.join(base_dir, filename)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=4)

        cache[filename] = entry
        

    def download_and_summarize_paper(self, entry, knowledge_type):
        """Download the paper PDF and generate a summary. Skip downloading if it already exists."""
        try:
            task_description = entry['Task Description'].replace(' ', '_')
            dataset_name = entry['Dataset Name'].replace(' ', '_')
            method = entry['Method'].replace(' ', '_')
            filename = f"{task_description}_{dataset_name}_{method}.json"

            if knowledge_type == 'experiment':
                cache = self.experiment_knowledge_cache
            elif knowledge_type == 'prior':
                cache = self.prior_knowledge_cache
            else:
                raise ValueError("Invalid knowledge type provided.")

            if filename in cache:
                # print(f"Summary already exists, skipping processing: {filename}")
                return cache[filename] 

            paper_url = entry['Paper Link']
            if "arxiv.org/abs" in paper_url:
                paper_url = paper_url.replace('abs', 'pdf')

            folder_path = Path(self.base_dir) / task_description / dataset_name
            folder_path.mkdir(parents=True, exist_ok=True)
            file_name = f"{method}.pdf"
            file_path = folder_path / file_name

            if not file_path.exists():
                response = requests.get(paper_url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                # print(f"Downloaded paper: {file_path}")
            else:
                # print(f"Paper already exists, skipping download: {file_path}")
                pass
            entry['Local Paper PDF Path'] = str(file_path)
            pdf_text = self.extract_text_from_pdf(str(file_path))
            prompt = """
            Please summarize the following paper content: 
            """ + pdf_text + """
            Focus specifically on the methods discussed in the article, 
            particularly the model design aspects. 
            Exclude the parts related to experiments, related work, or evaluation.
            """
            print(f"[Processing knowledge from OGB] Processing knowledge of method {entry['Method']}.")
            summary = llm_api.call_llm(prompt)
            entry['Paper Summary'] = summary['answer']
            self.save_to_knowledge_base(entry, knowledge_type)
            return entry
        except Exception as e:
            # print(f"[Processing knowledge from OGB] Failed to process the paper: {e}")
            return None

    
    def process_ogb_leaderboard(self, top_k=5, task_plan=None):
        """Process OGB leaderboard, download papers, and generate summaries, filter relevant entries based on task plan"""
        # Scrape OGB webpage and parse data

        leaderboard_data = []
        for url in self.ogb_urls:
            html_content = self.get_webpage_content(url)
            data = self.parse_ogb_leaderboard(html_content)
            leaderboard_data.extend(data)
            
        leaderboard_data = self.select_top_k_methods(leaderboard_data, k=top_k)

        # If a task plan is provided, use LLM to determine relevance
        if task_plan:
            relevant_data = []
            # for entry in tqdm(leaderboard_data, desc="Filtering"):
            for entry in leaderboard_data:
                relevance_prompt = self.generate_relevance_prompt(entry, task_plan)
                llm_response = llm_api.call_llm(relevance_prompt)
                time.sleep(0.5)
                if 'relevant' in llm_response['answer'].lower():
                    relevant_data.append(entry)
                    print(f"[Processing knowledge from OGB] Task {entry['Task Description']} and dataset {entry['Dataset Name']} of method {entry['Method']} is relevant.")
                else:
                    print(f"[Processing knowledge from OGB] Task {entry['Task Description']} and dataset {entry['Dataset Name']} method {entry['Method']} is not relevant.")
            leaderboard_data = relevant_data if relevant_data else [{}]
        else:
            leaderboard_data = leaderboard_data if leaderboard_data else [{}]
        if not leaderboard_data:
            return {}  

        updated_leaderboard_data = []
        # for entry in tqdm(leaderboard_data, desc="Summary"):
        for entry in leaderboard_data:
            filename = self.construct_filename(entry)

            if filename in self.prior_knowledge_cache:
                print(f"[Processing knowledge from OGB] Knowledge of method {entry['Method']} already exists.")
                continue  
            updated_entry = self.download_and_summarize_paper(entry, 'prior')
            if updated_entry is not None:
                updated_leaderboard_data.append(updated_entry)

        return updated_leaderboard_data

    def generate_relevance_prompt(self, entry, task_plan):
        """Generate a prompt to evaluate the relevance of the dataset to the task plan"""
        prompt = f"Evaluate the relevance of the following dataset to the task plan:\n"
        prompt += f"Task Description: {entry['Task Description']}\n"
        prompt += f"Dataset Name: {entry['Dataset Name']}\n"
        
        for key, value in task_plan.items():
            prompt += f"{key}: {value}\n"
        return prompt

    def construct_filename(self, entry):
        """Construct a filename based on the entry content"""
        task_description = entry['Task Description'].replace(' ', '_')
        dataset_name = entry['Dataset Name'].replace(' ', '_')
        method = entry['Method'].replace(' ', '_')
        return f"{task_description}_{dataset_name}_{method}.json"


    def generate_prompts_from_task_plan(self, task_plan):
        """Generate two prompts based on the task plan"""
        prompt_ogb = "Based on the following task plan, please select suitable OGB datasets and explain the reasons:\n"
        for key, value in task_plan.items():
            prompt_ogb += f"{key}: {value}\n"

        prompt_search = "Please generate a suitable search query based on the following task plan:\n"
        for key, value in task_plan.items():
            prompt_search += f"{key}: {value}\n"

        return prompt_ogb, prompt_search

    def parse_llm_response(self, response_text):
        """Parse the response from the LLM and extract the required datasets or query terms"""
        datasets = [s.strip() for s in response_text.replace('\n', ',').split(',') if s.strip()]
        return datasets

    def google_search(self, query, num_results=10):
        """Use Google Search API to perform a query and return the top k results containing title, snippet, and link"""
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cse_id,
            'num': num_results  # 返回结果数量
        }
        response = requests.get(base_url, params=params)
        items = response.json().get('items', [])
        results = []
        for item in items:
            results.append({
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'link': item.get('link')
            })
        return results

    def search_and_summarize(self, query, knowledge_type= 'prior'):
        """Search for information on Google based on the query and generate a summary"""
        search_results = self.google_search(query)
        if not search_results:
            print("No search results found.")
            return


        prompt = "Generate a summary based on the following search results:\n"
        for result in search_results:
            prompt += f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}\n\n"

        summary = llm_api.call_llm(prompt)
        knowledge_entry = {
            'Query': query,
            'Summary': summary['answer'],
            # 'Search Results': search_results  
        }

        filename = f"search_{query.replace(' ', '_')}.json"
        
        
        if knowledge_type == 'experiment':
            filepath = os.path.join(self.experiment_knowledge_base, filename)
        elif knowledge_type == 'prior':
            filepath = os.path.join(self.prior_knowledge_base, filename)
        # filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge_entry, f, ensure_ascii=False, indent=4)
        # self.cache[filename] = knowledge_entry
        
        self.save_to_knowledge_base(knowledge_entry)
        print(f"Saved search summary: {filepath}")


    def process_pyg_info(self, type='nn'):
        """Process Pyg information, download papers, and generate summaries"""
        pyg_info = self.parse_pyg_html(type)
        
        if type not in self.pyg_info:
            self.pyg_info[type] = {}
        
        
        # for entry in tqdm(pyg_info, desc="Summary"):
        for entry in pyg_info:
            name = entry['name'].replace(' ', '_')
            filename = f"{name}.json"

            if filename in self.pyg_info[type]:
                # print(f"Already processed, skipping: {filename}")
                print(f"[Processing knowledge from PyG] Knowledge of operation {entry['name']} already exists.")
                continue 


            try:
                update_entry = entry.copy()
                paper_url = update_entry['paper_link']
                if "arxiv.org/abs" in paper_url:
                    paper_url = paper_url.replace('abs', 'pdf')

                folder_path = Path(self.pyg_info_dir) / type 
                folder_path.mkdir(parents=True, exist_ok=True)
                file_name = f"{name}.pdf"
                file_path = folder_path / file_name

                if not file_path.exists():
                    response = requests.get(paper_url)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    # print(f"Downloaded paper: {file_path}")
                else:
                    # print(f"Paper already exists, skipping download: {file_path}")
                    pass

                update_entry['Local Paper PDF Path'] = str(file_path)

                pdf_text = self.extract_text_from_pdf(str(file_path))

                print(f"[Processing knowledge from PyG] Processing knowledge of operation {update_entry['name']}.")
                prompt = f"""Please summarize the following paper content. 
                        Focus specifically on the methods discussed in the article, particularly the model design aspects. 
                        Exclude the parts related to experiments, related work, or evaluation. 
                        Here is the text content of the paper:\n
                        """ + pdf_text
                
                
                summary = llm_api.call_llm(prompt)
                update_entry['Paper Summary'] = summary['answer']

                with open(str(file_path).replace(".pdf", ".json"), 'w', encoding='utf-8') as f:
                    json.dump(update_entry, f, ensure_ascii=False, indent=4)
                self.pyg_info[type][filename] = update_entry
                
            except Exception as e:
                # print(f"[Processing knowledge from PyG] Failed to process the paper: {e}")
                folder_path = Path(self.pyg_info_dir) / type 
                folder_path.mkdir(parents=True, exist_ok=True)
                file_name = f"{name}.pdf"
                file_path = folder_path / file_name
                with open(str(file_path).replace(".pdf", ".json"), 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=4)
                self.pyg_info[type][filename] = update_entry


            
    
    
    def parse_pyg_html(self, type='nn'):
        """
        Args:
            type (str, optional): _description_. Defaults to 'nn'.
                nn: torch_geometric.nn
                transforms: torch_geometric.transforms
            pyg_url (str, optional): _description_. Defaults to 'https://pytorch-geometric.readthedocs.io/en/latest/'.
        """
        url = self.pyg_url+ f'modules/{type}.html'
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        with open(type, 'w', encoding='utf-8') as file:
            file.write(soup.prettify())
        
        if type == 'nn':
            section = soup.find('section', id='convolutional-layers')
        elif type == 'transforms':
            section = soup.find('section', id='graph-transforms')
        
        pyg_info = []

        for row in section.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 2:
                layer_name = cells[0].text.strip()
                layer_description = cells[1].text.strip()
                layer_link = cells[0].find('a')['href'] if cells[0].find('a') else None
                
                external_link = cells[1].find('a')['href'] if cells[1].find('a') else None
                external_text = cells[1].find('a').text if cells[1].find('a') else None
                
                layer_info = {
                    "name": layer_name,
                    "description": layer_description,
                    "link": layer_link,
                    "paper_link": external_link,
                    "paper_name": external_text
                }
                pyg_info.append(layer_info)
                
        return pyg_info



    def search_with_task_plan(self, task_plan):
        """Generate search query based on task_plan, then perform search and summarize"""
        search_query = task_plan['Data'] + ' ' + task_plan['Learning_tasks_on_graph']
        # print(search_query)
        print("[Processing knowledge from search engine] Search information ")
        self.search_and_summarize(search_query)
    
    

    def save_pipeline_result(self, all_result, description="Pipeline result"):
        """Save the entire pipeline result to the experiment knowledge base"""
        
        path = all_result.get('path', '')
        model_type = None
        if './NODE_LEVEL/' in path:
            model_type = 'NODE_LEVEL'
        elif './GRAPH_LEVEL/' in path:
            model_type = 'GRAPH_LEVEL'
        elif './LINK_LEVEL/' in path:
            model_type = 'LINK_LEVEL'
        if model_type:
            background_file_path = os.path.join(self.base_dir, 'background', f'{model_type}.json')
            
            while True:
                current_path = os.getcwd()
                # print("Current working directory:", current_path)
                
                try:
                    with open(background_file_path, 'r', encoding='utf-8') as f:
                        background_data = json.load(f)
                        all_result['background_info'] = background_data
                        # print(f"File found and loaded successfully in {current_path}")
                        break  
                except OSError as e:
                    parent_directory = os.path.abspath(os.path.join(current_path, '..'))
                    
                    if current_path == parent_directory:
                        print("Reached the root directory, file not found.")
                        break
                    else:
                        os.chdir(parent_directory)

        
        filename = f"{description.replace(' ', '_')}_{int(time.time())}.json"  
        # filepath = os.path.join(self.experiment_knowledge_base, filename)

        current_directory = os.getcwd()
        
        # parent_directory = os.path.dirname(current_directory)

        filepath = os.path.join(current_directory, self.experiment_knowledge_base, filename)
        
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_result, f, ensure_ascii=False, indent=4)

        self.experiment_knowledge_cache[filename] = all_result
        # print(f"\n\n***Pipeline result saved: {filepath}***\n\n")
        
    def summarize_pdf(self, pdf_directory, output_directory):
        """
        Summarizes PDF files located in the given directory.
        
        Args:
            pdf_directory (str): The directory containing the PDF files.
            output_directory (str): The directory where the summary JSON files will be saved.
        """
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Get list of PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        existing_summaries = set(f.replace('.json', '') for f in os.listdir(output_directory) if f.endswith('.json'))

        for pdf_file in tqdm(pdf_files):
            paper_name = pdf_file.replace('.pdf', '')

            # Skip if summary already exists
            if paper_name in existing_summaries:
                print(f"Summary for '{paper_name}' already exists. Skipping.")
                continue

            pdf_path = os.path.join(pdf_directory, pdf_file)

            # Extract text from PDF
            try:
                text = self.extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            # Create prompt for LLM
            prompt = f"""
            Please summarize the content of the following paper 
            and provide the response in the form of a Python dictionary with the following keys:
            1. paper_name: The name of the paper.
            2. method_name: The name of the method or model discussed.
            3. method_summary: A summary of the methods and model design aspects. Do not include any information about experiments or related work.
            4. experiment_summary: A summary of the experiment part.

            Here is the text of the paper:
            {text}
            """
            
            max_retries = 10
            for attempt in range(max_retries):
            
                response = llm_api.call_llm(prompt)
                
                # print(response)
                answer = response['answer'].replace('\n', '').replace('```', '').replace('\\', '')
                match = re.search(r'\{\s*.*?\s*\}', answer, re.DOTALL)
                
                # print(answer)


                if match:
                    json_str = match.group()
                    try:
                        # print("json_str done")
                        json_data = json.loads(json_str)
                        # print("json_data done")
                        # print(json_data)
                        # time.sleep(1000)
                        summary = {
                            'paper_name': json_data.get('paper_name', ''),
                            'method_name': json_data.get('method_name', ''),
                            'method_summary': json_data.get('method_summary', ''),
                            'experiment_summary': json_data.get('experiment_summary', '')
                        }
                        # print("summary:",summary)

                        output_file_path = os.path.join(output_directory, f"{pdf_file.replace('.pdf', '.json')}")
                        print("output_file_path:",output_file_path)
                        with open(output_file_path, 'w', encoding='utf-8') as json_file:
                            json.dump(summary, json_file, ensure_ascii=False, indent=4)
                        print(f"Summary for '{pdf_file}' saved to '{output_file_path}'.")
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
