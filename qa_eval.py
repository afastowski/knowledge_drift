from datasets import load_dataset
from openai import OpenAI
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, MistralForCausalLM
from huggingface_hub import login
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Evaluating baseline QA performance.")
parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo-0125", help="LLM to query.")
args = parser.parse_args()

num_samples = 1000

if "gpt" in args.model:
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    client = OpenAI(
    organization = os.environ['OPENAI_ORG'],
    api_key = os.environ['OPENAI_KEY']
    )

elif "llama" in args.model:
    login(token = os.environ['HF_TOKEN'])
    model_id = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")

elif "mistral" in args.model:
    login(token = os.environ['HF_TOKEN'])
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MistralForCausalLM.from_pretrained(model_id, device_map="auto")


def generate_chat_completion_gpt(message_dict, model=args.model, max_tokens=None, log_probs=True):
    results = client.chat.completions.create(
        model=model,
        messages=message_dict,
        logprobs=log_probs,
        seed=42,
        temperature=0
    )
    return results

def generate_chat_completion_llama(message_dict):
    input_ids = tokenizer.apply_chat_template(
        message_dict,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        temperature=0.01
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def generate_chat_completion_mistral(message_dict):
    model.config.pad_token_id = model.config.eos_token_id
    input_ids = tokenizer.apply_chat_template(
        message_dict,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        # if you don't set max_new_tokens for mistral, it throws an error
        max_new_tokens=10,
        temperature=0.01,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def eval():
    out_file = f"data/correctly_answered_1000_{args.model}.json"
    with open(out_file, 'a') as f:
        f.write('[')

    current_id = 0
    dataset = load_dataset("json", data_files="data/triviaqa_1000.json")
    correct = 0
    for x, sample in enumerate(dataset["train"]):
        question = sample["question"]
        true_answer = sample["true_answer"]
        false_info_context = sample["context"]

        if "gpt" in args.model:
            message_dict = [{
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{question} Respond with the exact answer only."
            }]

            API_RESPONSE = generate_chat_completion_gpt(message_dict)
            model_answer = API_RESPONSE.choices[0].message.content

        elif "llama" in args.model:
            message_dict = [
                {"role": "system", "content": "You are a helpful assistant who responds as shortly as possible. Your responses are only 1-3 words long."},
                {"role": "user", "content": f"{question} Respond with the exact answer only."},
            ]

            model_answer = generate_chat_completion_llama(message_dict)
        elif "mistral" in args.model:
            message_dict = [
                {"role": "user", "content": "You are a helpful assistant who responds as shortly as possible. Your responses are only 1-3 words long."},
                {"role": "assistant", "content": "Understood, I will respond with a fact only."},
                {"role": "user", "content": f"{question} Respond with the exact answer only."}
            ]
            model_answer = generate_chat_completion_mistral(message_dict)
            print(model_answer)


        # model often produces slightly more detailed answers, which are still the same answer, e.g. "Chicago" vs "Chicago, Illinois"
        if true_answer.lower() in model_answer.lower():
            correct += 1

            entry = {
            "id": current_id,
            "question": question,
            "answer": true_answer,
            "false_context": false_info_context,
        }
            current_id += 1

            with open(out_file, 'a') as f:
                f.write(json.dumps(entry, indent=4))
                f.write(',\n')

    with open(out_file, 'a') as f:
        f.write(']')

    acc = (correct / num_samples) * 100
    print(f"Model: {args.model}")
    print(f"Accuracy: {acc}%")

eval()