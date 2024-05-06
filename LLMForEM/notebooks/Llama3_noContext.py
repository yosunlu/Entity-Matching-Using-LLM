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
    'wdcproducts-80cc-seen-sampled-250-gs-2_domain-complex-force'
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
        
        prompts = []
        
        if attribute == 'default':
            for example in task_dict['examples']:
                messages = [
                    {"role": "system", "content": "You are an entity matcher"},
                ]
                messages.append({"role" : "user", "content" : f"{task_dict['task_prefix']}{example['input']}"})
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                prompts.append(prompt)
                
        targets = [example['target_scores'] for example in task_dict['examples']]
        
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompts,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
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

            processed_pred = outputs[i][0]['generated_text'].replace(f'{prompts[i]}', '')
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
        if not os.path.exists('./prompts/'):
            os.makedirs('./prompts/')
        with open(f'./prompts/{task}_{attribute}_{model_name}_run-1.pickle', 'wb') as handle:
            pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save the answers
        if not os.path.exists('./answers/'):
            os.makedirs('./answers/')
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

        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        with open(f"./results/result_{task}_{attribute}_{model_name}_run-1.json", "w") as outfile:
            json.dump(results, outfile, indent=2)


