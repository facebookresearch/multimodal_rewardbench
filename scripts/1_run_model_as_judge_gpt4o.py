# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
import os
import time
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool

import base64
from mimetypes import guess_type

# Load model
from openai import AzureOpenAI
endpoint = 'azure-services-fair-openai1-westus.azure-api.net'
API_key=os.getenv("AZ_OAI_API")

client = AzureOpenAI(
    api_version="2024-06-01",
    api_key=API_key,
    azure_endpoint=f'https://{endpoint}'
)


# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# Function to prompt the model
def prompt(ex, max_try=10):
    image_folder = ex['image_folder']
    image_path = f"{image_folder}/{ex['Image']}"
    image_url = local_image_to_data_url(image_path)
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text":  ex['Text']
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                } 
            ]
        }
    ]
    count = 1
    while count < max_try:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
                max_tokens=1024,
            )
            out = completion.to_dict()['choices'][0]['message']['content'].strip()
            return out
        except Exception as e:
            print('Exception:', e)
            count += 1
            time.sleep(2) 
    return "None"


judge_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""


def prepare_judge_example(ex):
    return {
        'ID': ex['ID'],
        'Text': judge_prompt.format(
            question=ex['Text'],
            answer_a=ex['Output1'],
            answer_b=ex['Output2'],
        ),
        'Image': ex['Image'],
        'Label': 'B' if ex['Better'] == 'Output2' else 'A',
        'Meta': ex,
    }


def main(args):
    exs = []
    for ex in json.load(open(os.path.expanduser(args.question_file), "r")):
        ex = prepare_judge_example(ex)
        ex['image_folder'] = args.image_folder
        for k in range(args.num_samples):
            ex = deepcopy(ex)
            exs.append(ex)
    
    with Pool(30) as p:
        outs = list(tqdm(p.imap(prompt, exs), total=len(exs), desc='Running model'))
        assert len(outs) == len(exs)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for i in range(0, len(exs), args.num_samples):
            ex = exs[i]
            outputs = outs[i : i + args.num_samples]
            if args.num_samples == 1:
                ex["output"] = outputs[0]
            else:
                ex["outputs"] = outputs
            _ = ex.pop('image_folder')
            ans_file.write(json.dumps(ex) + "\n")
            ans_file.flush()
    print('Done.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="data")
    parser.add_argument("--question-file", type=str, default="data/all_data.json")
    parser.add_argument("--answers-file", type=str, default="outputs/gpt4o.jsonl")
    parser.add_argument('-n', "--num-samples", type=int, default=1)
    args = parser.parse_args()

    main(args)

"""
cd <repo>
conda activate openai

python scripts/1_run_model_as_judge_gpt4o.py 
"""