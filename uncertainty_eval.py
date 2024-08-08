from datasets import load_dataset
import numpy as np
from collections import defaultdict
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description="Evaluating uncertainty scores.")
parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo-0125", help="LLM which scores to evaluate")
parser.add_argument("-f", "--falseinfo", action="store_true", help="Whether to add false info into the prompt.")
parser.add_argument("-fn", "--falseinfonum", type=int, default=1, help="How often to prepend the false information.")
parser.add_argument("-r", "--randominfo", action="store_true", help="Whether to add random info into the prompt.")
parser.add_argument("-v", "--version", type=int, default=1, help="Prompt v1 or prompt v2.")
args = parser.parse_args()

mode = ""
if args.falseinfo:
    mode = "false"
    if args.falseinfonum > 1:
        mode = f"false_x{args.falseinfonum}"
elif args.randominfo:
    mode = "random"
else:
    mode = "baseline"

ds_uncertainties = load_dataset("json", data_files=f"scores_1000/{args.model}/uncertainties_{mode}_prompt_v{args.version}.json")
uncertainties = defaultdict(list)


for sample in ds_uncertainties["train"]:
    ae = np.mean(sample["uncertainty"]["ae"])
    ppl = np.mean(sample["uncertainty"]["ppl"])
    ap = np.mean(sample["uncertainty"]["ap"])

    counter = Counter(sample["model_answer"])
    model_answer, frequency = counter.most_common(1)[0]
    uncertainties["model_answer"].append(model_answer)
    uncertainties["correct_answer"].append(sample["answer"])

    uncertainties["ae"].append(ae)
    uncertainties["ppl"].append(ppl)
    uncertainties["ap"].append(ap)

num_results = len(uncertainties["ae"])
ae = np.mean(uncertainties["ae"])
ae_std_err =np.std(uncertainties["ae"]) / np.sqrt(num_results)
ppl = np.mean(uncertainties["ppl"])
ppl_std_err = np.std(uncertainties["ppl"]) / np.sqrt(num_results)
ap = np.mean(uncertainties["ap"])
ap_std_err = np.std(uncertainties["ap"]) / np.sqrt(num_results)

## Measure Answer Accuracy.

correct = 0

correct_answers = defaultdict(list)
incorrect_answers = defaultdict(list)

for i, example in enumerate(uncertainties["ae"]):
    if uncertainties["correct_answer"][i].lower() in uncertainties["model_answer"][i].lower():
        correct += 1
        
        correct_answers["ae"].append(uncertainties["ae"][i])
        correct_answers["ppl"].append(uncertainties["ppl"][i])
        correct_answers["ap"].append(uncertainties["ap"][i])

    # if false prompt changed the answer to a wrong one
    else:
        #print(uncertainties["correct_answer"][i], "---", uncertainties["model_answer"][i])
        incorrect_answers["ae"].append(uncertainties["ae"][i])
        incorrect_answers["ppl"].append(uncertainties["ppl"][i])
        incorrect_answers["ap"].append(uncertainties["ap"][i])

print("Overall Accuracy: ", round(correct/num_results, 4))
print()

##################
print("Correct Answers Metrics:")

correct_num_results = len(correct_answers["ae"])

correct_ae = np.mean(correct_answers["ae"])
correct_ae_std_err =np.std(correct_answers["ae"]) / np.sqrt(correct_num_results)

correct_ppl = np.mean(correct_answers["ppl"])
correct_ppl_std_err = np.std(correct_answers["ppl"]) / np.sqrt(correct_num_results)

correct_ap = np.mean(correct_answers["ap"])
correct_ap_std_err = np.std(correct_answers["ap"]) / np.sqrt(correct_num_results)

print("Correct Ratio: ", correct_num_results / num_results)
print("Entropy: ", f"${round(correct_ae,2)}^{{\pm{round(correct_ae_std_err,3)}}}$")
print("PPL: ", f"${round(correct_ppl,2)}^{{\pm{round(correct_ppl_std_err,3)}}}$")
print("Prob: ", f"${round(correct_ap,2)}^{{\pm{round(correct_ap_std_err,3)}}}$")
print()

###################
print("Incorrect Answers Metrics:")

incorrect_num_results = len(incorrect_answers["ae"])

incorrect_ae = np.mean(incorrect_answers["ae"])
incorrect_ae_std_err =np.std(incorrect_answers["ae"]) / np.sqrt(incorrect_num_results)

incorrect_ppl = np.mean(incorrect_answers["ppl"])
incorrect_ppl_std_err = np.std(incorrect_answers["ppl"]) / np.sqrt(incorrect_num_results)

incorrect_ap = np.mean(incorrect_answers["ap"])
incorrect_ap_std_err = np.std(incorrect_answers["ap"]) / np.sqrt(incorrect_num_results)

print("Incorrect Ratio: ", incorrect_num_results / num_results)
print("Entropy: ", f"${round(incorrect_ae,2)}^{{\pm{round(incorrect_ae_std_err,3)}}}$")
print("PPL: ", f"${round(incorrect_ppl,2)}^{{\pm{round(incorrect_ppl_std_err,3)}}}$")
print("Prob: ", f"${round(incorrect_ap,2)}^{{\pm{round(incorrect_ap_std_err,3)}}}$")
