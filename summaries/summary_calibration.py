import ast
import json
import os
import sys

sys.path.append("..") 

from collections import defaultdict

from datasets import Dataset
from vllm import LLM, SamplingParams

from args import parser
from sentence_utils import (all_pos_filter, get_basename, llm_based_adj_filter,
                            remove_mutual_information, rule_based_adj_filter)

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")

	processed_summary = remove_mutual_information(args.save_summary_dir, args.probing_dataset, args.concept_set_file, args.target_name, args.sentence_transformer, device=args.device, K=args.K)

	processed_summary = rule_based_adj_filter(processed_summary, args.target_name)
	
	prompt_template = "<s>[INST] <<SYS>>\n\
	<</SYS>>\n\n \
	Can the adjective '{}' be used to describe the tone, emotion, or acoustic features of audio, music, or any other form of sound?\n \
	Answer(yes or no):\n\
	Reason:\
	[/INST]"

	with open(f"all_adj_{args.target_name}.txt") as f: 
		words = ast.literal_eval(f.read())

	dataset = defaultdict(list)
	for word in words:
		dataset["word"].append(word)
		dataset["text"].append(prompt_template.format(word))
	dataset = Dataset.from_dict(dataset)

	llm = LLM(model=args.llm)

	sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, max_tokens=args.max_tokens)

	prompts = [prompt_template.format(w) for w in words]

	outputs = llm.generate(prompts, sampling_params)

	results = []
	for idx, output in enumerate(outputs):
		prompt = output.prompt
		word = dataset[idx]["word"]
		generated_text = output.outputs[0].text
		results.append({"word": word, "response": generated_text})

	acoustic_words_file = f"acoustic_adj_{args.target_name}.json"
	with open(acoustic_words_file, "w") as f: 
		json.dump(results, f, indent=2)

	processed_summary = llm_based_adj_filter(processed_summary, acoustic_words_file)

	processed_summary = all_pos_filter(processed_summary)

	with open(os.path.join(args.save_summary_dir, f"calibration_{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}_top{args.K}.json"), "w") as f:
		json.dump(processed_summary, f, indent=2)
