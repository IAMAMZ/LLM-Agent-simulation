# import os
# from huggingface_hub import hf_hub_download

model_id = "microsoft/Phi-4-mini-instruct"


# file_names = [
#     "CODE_OF_CONDUCT.md",
#     "LICENSE",
#     "NOTICE.md",
#     "README.md",
#     "SECURITY.md",
#     "added_tokens.json",
#     "config.json",
#     "configuration_phi3.py",
#     "generation_config.json",
#     "merges.txt",
#     "model-00001-of-00002.safetensors",
#     "model-00002-of-00002.safetensors",
#     "model.safetensors.index.json",
#     "modeling_phi3.py",
#     "sample_finetune.py",
#     "special_tokens_map.json",
#     "tokenizer.json",
#     "tokenizer_config.json",
#     "vocab.json"
# ]

# for filename in file_names:
#         downloaded_model_path = hf_hub_download(
#                     repo_id=model_id,
#                     filename=filename,
#         )
#         print(downloaded_model_path)


import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # "cpu" or "auto" or "cuda:0" for cuda device 0, 1, 2, 3 etc. if you have multiple GPUs
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

#gotta have a tokenizer for each model otherwise the token mappings won't match
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
#lower temperature generally more predictable results, you can experiment with this
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 

while (True):

    # "What do you think of Tom Moreno?"
    prompt = input("Enter a prompt: ")
    if prompt == "quit":
        break
    
    # system - context for the AI (sort of like a role), user - a prompt
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant."},
    #     {"role": "user", "content": "What do you know about Lakehead University?"},
    # ]

    # from LLM agents paper - using "video game" or "simulation" to prevent it from saying "...as an AI I don't have opinions"
    # use something like "John Lin Thoughts:" for an internal monologue
    # use something like "Tom Moreno: Hi John!" to act as another agent
    messages = [
        {"role": "system", "content": "You are John Lin, a friendly townsperson in a video game."},
        {"role": "system", "content": '''You, John Lin, are a pharmacy shopkeeper at the Willow
    Market and Pharmacy who loves to help people. He
    is always looking for ways to make the process
    of getting medication easier for his customers;
    John Lin is living with his wife, Mei Lin, who
    is a professor, and son, Eddy Lin, who is
    a student studying music; John Lin loves
    his family very much; John Lin has known the
    couple next-door, Sam Moore and Jennifer Moore,
    for years; John Lin thinks Sam Moore is a
    kind and nice man; John Lin and Tom Moreno
    are colleagues at The Willows Market and Pharmacy;
    John Lin and Tom Moreno are friends and like to
    discuss local politics together.'''},
        {"role": "user", "content": prompt},
    ]

    time1 = int(round(time.time() * 1000))

    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])

    time2 = int(round(time.time() * 1000))
    print("Generation time: " + str(time2 - time1))
