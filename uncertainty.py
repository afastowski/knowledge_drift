from datasets import load_dataset
from openai import OpenAI
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
from collections import defaultdict
import json
import random
from uncertainty_metrics import *
import argparse

parser = argparse.ArgumentParser(description="Running uncertainty metrics.")
parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo-0125", help="LLM to query.")
parser.add_argument("-f", "--falseinfo", action="store_true", help="Whether to add false info into the prompt.")
parser.add_argument("-fn", "--falseinfonum", type=int, default=1, help="How often to prepend the false information.")
parser.add_argument("-r", "--randominfo", action="store_true", help="Whether to add random info into the prompt.")
parser.add_argument("-v", "--version", type=int, default=1, help="Prompt v1 or prompt v2.")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU to use.")

args = parser.parse_args()

# number of candidate tokens per position (for calculating entropy at each answer position)
num_candidates = 10
mode = ""

if args.falseinfo:
    mode = "false"
    if args.falseinfonum > 1:
        mode = f"false_x{args.falseinfonum}"
elif args.randominfo:
    mode = "random"
else:
    mode = "baseline"


out_file = f"scores_1000/{args.model}/uncertainties_{mode}_prompt_v{args.version}.json"

if "llama" in args.model:
    login(token = os.environ['HF_TOKEN'])
    model_id = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, device_map=f"cuda:{args.gpu}")

elif "mistral" in args.model:
    login(token = os.environ['HF_TOKEN'])
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MistralForCausalLM.from_pretrained(model_id, device_map=f"cuda:{args.gpu}")

# apparently, you can't make openai models fully deterministic, hence we're getting slightly different logprobs each run.
# -> average over n runs
def generate_chat_completion_gpt(client, messages, model=args.model, max_tokens=None, log_probs=True, top_logprobs=num_candidates):
    client = client
    results = client.chat.completions.create(
        model=model,
        messages=messages,
        logprobs=log_probs,
        top_logprobs=top_logprobs,
        seed=42,
        temperature=0
    )
    return results


def generate_chat_completion_llama(message_dict):
    #ensure different output for each of the 10 runs, to later average over
    seed = random.randint(0, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    input_ids = tokenizer.apply_chat_template(
        message_dict,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(f"cuda:{args.gpu}")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        output_logits=True, 
        return_dict_in_generate=True
    )
    response = outputs[0][0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    logits = outputs.logits
    return logits, response


def generate_chat_completion_mistral(message_dict):
    seed = random.randint(0, 100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.config.pad_token_id = model.config.eos_token_id
    input_ids = tokenizer.apply_chat_template(
        message_dict,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids,
        # if you don't set max_new_tokens for mistral, it throws an error
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        output_logits=True,
        return_dict_in_generate=True
    )
    response = outputs[0][0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    logits = outputs.logits
    return logits, response


def calc_uncertainties():

    with open(out_file, 'a') as f:
        f.write('[')

    dataset = load_dataset("json", data_files=f"data/correctly_answered_1000_{args.model}.json")
    if args.randominfo:
        contexts = dataset["train"]["false_context"]
        random.seed(42)
        random.shuffle(contexts)

    current_id = 1

    for x, sample in enumerate(dataset["train"]):
        print("Current sample: ", x)
        question = sample["question"]
        true_answer = sample["answer"]
        false_info = sample["false_context"]

        prompt_first_part = ""
        base_prompt = ""
        prompt = ""

        if args.version == 2:
            base_prompt = f"{question} Respond with the true, exact answer only."
        else: 
            base_prompt = f"{question} Respond with the exact answer only."

        if args.falseinfo:
            prompt_first_part += f"{false_info} " * args.falseinfonum
            prompt = prompt_first_part + base_prompt
        elif args.randominfo:
            random_context = contexts[x]
            prompt_first_part += f"{random_context} "
            prompt = prompt_first_part + base_prompt
        else:
            prompt = base_prompt

        print(prompt)

        if "gpt" in args.model:
            message_dict = [
                {"role": "system", "content": "You are a helpful assistant who responds as shortly as possible. Your responses are only 1-3 words long."},
                {"role": "user", "content": prompt}
            ]
        
        elif "llama" in args.model:
            message_dict = [
                {"role": "system", "content": "You are a helpful assistant who responds as shortly as possible. Your responses are only 1-3 words long."},
                {"role": "user", "content": prompt},
            ]

        elif "mistral" in args.model:
            message_dict = [
                {"role": "user", "content": "You are a helpful assistant who responds as shortly as possible. Your responses are only 1-3 words long."},
                {"role": "assistant", "content": "Understood, I will respond with a fact only."},
                {"role": "user", "content": prompt}
            ]

        uc_results = defaultdict(list)
        model_answers = []
        num_runs = 10
        for _ in range(num_runs):

            if "gpt" in args.model:
            # instantiate again every time do get same effect as new run (will produce slightly different result)
                client = OpenAI(
                    organization = os.environ['OPENAI_ORG'],
                    api_key = os.environ['OPENAI_KEY']
                    )
                API_RESPONSE = generate_chat_completion_gpt(client, message_dict)
                model_answer = API_RESPONSE.choices[0].message.content
                answer_entropy = get_avg_entropy_gpt(API_RESPONSE)
                answer_perplexity = get_perplexity_gpt(API_RESPONSE)
                answer_probability = get_avg_probability_gpt(API_RESPONSE)

            elif "llama" in args.model:
                logits, model_answer = generate_chat_completion_llama(message_dict)
                answer_entropy = get_avg_entropy_hf(logits)
                answer_perplexity = get_perplexity_hf(logits)
                answer_probability = get_avg_probability_hf(logits)

            elif "mistral" in args.model:
                logits, model_answer = generate_chat_completion_mistral(message_dict)
                answer_entropy = get_avg_entropy_hf(logits)
                answer_perplexity = get_perplexity_hf(logits)
                answer_probability = get_avg_probability_hf(logits)

            model_answers.append(model_answer)

            uc_results["ae"].append(answer_entropy)
            uc_results["ppl"].append(answer_perplexity)
            uc_results["ap"].append(answer_probability)

        entry = {
            "id": current_id,
            "question": question,
            "false_info": false_info,
            "answer": true_answer,
            "model_answer": model_answers,
            "uncertainty": uc_results,
        }
        #print(entry)
        with open(out_file, 'a') as f:
                f.write(json.dumps(entry, indent=4))
                f.write(',\n')

        current_id += 1

    with open(out_file, 'a') as f:
        f.write(']')


calc_uncertainties()
