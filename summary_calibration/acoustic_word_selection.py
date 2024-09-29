import os
import torch
import json
import ast
import re
from collections import defaultdict

from args import parser
from vllm import LLM, SamplingParams
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):

	prompt_template = "<s>[INST] <<SYS>>\n\
<</SYS>>\n\n \
Can the adjective '{}' be used to describe the tone, emotion, or acoustic features of audio, music, or any other form of sound?\n \
Answer(yes or no):\n\
Reason:\
[/INST]"

	with open("results/all_adj_beats_unfreeze.txt") as f: 
		words = ast.literal_eval(f.read())

	dataset = defaultdict(list)
	for word in words:
		dataset["word"].append(word)
		dataset["text"].append(prompt_template.format(word))
	dataset = Dataset.from_dict(dataset)


	llm = LLM(model="meta-llama/Llama-2-13b-hf")

	sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, max_tokens=args.max_tokens)

	prompts = [prompt_template.format(w) for w in words]

	outputs = llm.generate(prompts, sampling_params)

	results = []
	for idx, output in enumerate(outputs):
		prompt = output.prompt
		word = dataset[idx]["word"]
		generated_text = output.outputs[0].text
		print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
		results.append({"word": word, "response": generated_text})

	with open("results/all_acoustic_adj_beats_unfreeze.json", "w") as f: 
		json.dump(results, f, indent=2)

if __name__ == "__main__":
	
	args = parser.parse_args()
	main()
