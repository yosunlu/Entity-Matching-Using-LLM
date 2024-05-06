# %%
from tqdm.notebook import tqdm

import json
import pickle
import string

from langchain import PromptTemplate

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import transformers
import torch

from pdb import set_trace

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

# %%
tasks = [
    'wdcproducts-80cc-seen-sampled-250-gs-2_domain-complex-force',
    ]

# %%
model_path = "/nobackup/huidang58kg/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

#model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# %%
attributes = ['default']

for task in tasks:
    for attribute in attributes:
        
        
        # open the JSON file in read mode
        with open(f'Entity-Matching-Using-LLM/LLMForEM/tasks/{task}.json', 'r') as f:
            # load the JSON data from the file and convert it into a dictionary
            task_dict = json.load(f)
        

        # Create LangChain PromptTemplate

        template = """{task_prefix}{input_string}"""
        prompt = PromptTemplate(
                template=template,
                input_variables=['task_prefix', 'input_string']
        )
        
        prompts = []
        
        
        if attribute == 'default':
            for example in task_dict['examples']:
                
                inputs_split = example['input'].split('\n\n')
                matches = inputs_split[0].split('\n')[1:]
                non_matches = inputs_split[1].split('\n')[1:]
                task_full = inputs_split[2]
                question = task_full.split('\n')[0]
                
                context_prompts = []

                for match in matches:
                    match = match.replace('{', '').replace('}', '')

                    example_human = f'### User:\n{question}\n{match}'
                    context_prompts.append(example_human)
                    example_ai = f'### Assistant:\nYes.'
                    context_prompts.append(example_ai)

                for non_match in non_matches:
                    non_match = non_match.replace('{', '').replace('}', '')

                    example_human = f'### User:\n{question}\n{non_match}'
                    context_prompts.append(example_human)
                    example_ai = f'### Assistant:\nNo.'
                    context_prompts.append(example_ai)        

                human_message_prompt = f'### User:\n{task_full}'

                context_prompts.append(human_message_prompt)

                chat_prompt = '\n\n'.join(context_prompts)

                text_prompt = f"{chat_prompt}\n\n### Assistant:\n"

                prompts.append(text_prompt)
                
        targets = [example['target_scores'] for example in task_dict['examples']]
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        sequences = pipeline(
            prompts,
            use_cache=True,
            max_new_tokens=100
        )
                
        # Do some data wrangling to format target and preds to match squad V2
        answers = []
        predictions = []
        truth = []
        unclear_answers = 0
        num_long_answers = 0
        for i in range(len(targets)):
            if targets[i]['Yes'] == 1:
                truth.append(1)
            else:
                truth.append(0)

            processed_pred = sequences[i][0]['generated_text'].replace(f'{prompts[i]}', '')
            answers.append(processed_pred)            
                    
            # handle yes/no answers
            
            processed_pred = processed_pred.strip().translate(str.maketrans('', '', string.punctuation)).lower()

            if processed_pred != 'yes' and processed_pred != 'no':
                print(f'Overlong Answer: {processed_pred}')
                num_long_answers += 1
            if 'yes' in processed_pred:
                processed_pred = 'yes'
            elif 'no' in processed_pred:
                processed_pred = 'no'
            else:
                processed_pred = 'no'
                unclear_answers += 1

            if processed_pred == 'yes':
                predictions.append(1)
            elif processed_pred == 'no':
                predictions.append(0)
        
        # save the prompts
        with open(f'./prompts/{task}_{attribute}_{model_name}_run-1.pickle', 'wb') as handle:
            pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save the answers
        with open(f'./answers/{task}_{attribute}_{model_name}_run-1.pickle', 'wb') as handle:
            pickle.dump(answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        precision = precision_score(truth, predictions)
        recall = recall_score(truth, predictions)
        f1 = f1_score(truth, predictions)
        accuracy = accuracy_score(truth, predictions)
        

        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

        with open(f"./results/result_{task}_{attribute}_{model_name}_run-1.json", "w") as outfile:
            json.dump(results, outfile, indent=2)


