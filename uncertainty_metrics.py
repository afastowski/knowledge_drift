import numpy as np
import torch

## Uncertainty: implements three metrics: 1) Average Entropy, 2) Perplexity, 3) Average Probability

def get_avg_entropy_gpt(API_RESPONSE):
    # get number of tokens in produced response
    num_produced_tokens = len(API_RESPONSE.choices[0].logprobs.content)
    # for each of those: top_logprobs = API_RESPONSE.choices[0].logprobs.content[i].top_logprobs where i is the current token position
    sum_all_entropies = 0
    
    for i in range(num_produced_tokens):
        entropy_current_position = 0
        # .content[i] -> token at position i in the response
        top_logprobs = API_RESPONSE.choices[0].logprobs.content[i].top_logprobs

        for logprob in top_logprobs:
            linear_probability = np.exp(logprob.logprob)
            log_probability = np.log2(linear_probability)
            entropy_current_position += linear_probability * log_probability

        sum_all_entropies += -(entropy_current_position)

    answer_entropy = sum_all_entropies / num_produced_tokens
    return answer_entropy

def get_avg_entropy_hf(logits):
    k = 10
    num_produced_tokens = len(logits) - 1 #ignore <\s> token at the end of the generation
    sum_all_entropies = 0
    
    for i in range(num_produced_tokens):
        entropy_current_position = 0
        probabilities = torch.log_softmax(logits[i], dim=-1).cpu()
        top_logprobs, _ = torch.topk(probabilities, k)

        for logprob in top_logprobs[0]:
            linear_probability = np.exp(logprob)
            if torch.isinf(logprob):
                logprob = torch.tensor(0)
            entropy_current_position += linear_probability * logprob

        sum_all_entropies += -(entropy_current_position)
    answer_entropy = sum_all_entropies / num_produced_tokens
    return answer_entropy.item()


def get_perplexity_gpt(API_RESPONSE):
    num_produced_tokens = len(API_RESPONSE.choices[0].logprobs.content)
    nll = []
    for i in range(num_produced_tokens):
        nll.append(API_RESPONSE.choices[0].logprobs.content[i].logprob)
    avg_nll = np.mean(nll)
    ppl = np.exp(-avg_nll)
    return ppl

def get_perplexity_hf(logits):
    num_produced_tokens = len(logits) - 1 #ignore <\s> token at the end of the generation
    nll = []
    for i in range(num_produced_tokens):
        probabilities = torch.log_softmax(logits[i][0], dim=-1).cpu()
        top_logprobs, _ = torch.topk(probabilities, 3)
        top_logprob = top_logprobs[0]
        nll.append(top_logprob.cpu())
    avg_nll = np.mean(nll)
    ppl = np.exp(-avg_nll)
    return ppl.item()


def get_avg_probability_gpt(API_RESPONSE):
    num_produced_tokens = len(API_RESPONSE.choices[0].logprobs.content)
    sum_linear_probs = []

    for i in range(num_produced_tokens):
        logprob = API_RESPONSE.choices[0].logprobs.content[i].logprob
        linear_probability = np.exp(logprob)
        sum_linear_probs.append(linear_probability)
    return np.mean(sum_linear_probs)


def get_avg_probability_hf(logits):
    num_produced_tokens = len(logits) - 1 #ignore <\s> token at the end of the generation
    sum_linear_probs = []

    for i in range(num_produced_tokens):
        probabilities = torch.log_softmax(logits[i], dim=-1).cpu()
        top_logprobs, _ = torch.topk(probabilities, 3)
        linear_probability_top_token = np.exp(top_logprobs[0][0])
        sum_linear_probs.append(linear_probability_top_token)
    return np.mean(sum_linear_probs).item()
