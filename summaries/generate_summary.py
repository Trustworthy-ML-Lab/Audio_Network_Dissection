import json
import os
import sys

sys.path.append("..")

import nltk
from vllm import LLM, SamplingParams

nltk.download("punkt")

from args import parser
from data_utils import get_description_dataset
from sentence_utils import get_basename

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")
	prompt_template = \
	"<s>[INST] <<SYS>>\n \
	Here are descriptions of some audio clips. Please summarize these descriptions by identifying their commonalities. \
	<</SYS>>\n\n  \
	{}  \
	[/INST]"

	args.target_layers = args.target_layers.split(",")
	args.save_activation_dir = os.path.join(args.save_activation_dir, f'{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}')


	dataset = get_description_dataset(args.audio_description_dir, args.save_activation_dir, args.probing_dataset, args.concept_set_file, args.target_name, args.target_layers, args.network_class_file, prompt_template, discriminative_type=args.discriminative_type, K=args.K)

	llm = LLM(model=args.llm, tensor_parallel_size=args.num_of_gpus, gpu_memory_utilization=0.9)

	sampling_params = SamplingParams(top_p=1, temperature=1, max_tokens=1024)

	inputs = [line["text"] for line in dataset]
	results = []
	outputs = llm.generate(inputs, sampling_params)

	for input, output in zip(dataset, outputs):
		target_layer = input["target_layer"]
		neuron_id = input["neuron_id"]
		label = input["label"]
		raw_inputs = input["raw_text"]
		print(f'-------- {target_layer} # {neuron_id}  --------')
		response = output.outputs[0].text
		results.append({"summary": response,
						"inputs": raw_inputs,
						"target_layer": target_layer,
						"neuron_id": neuron_id,
						"neuron_label": label,
						"audio_labels": input["audio_labels"],
						"discriminative_type": args.discriminative_type})

	if not os.path.exists(args.save_summary_dir):
		os.makedirs(args.save_summary_dir)

	# Note that summary file will be overridden when applying different target layers
	summary_file = os.path.join(args.save_summary_dir, f'{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}_{args.discriminative_type}_top{args.K}.json')
	
	with open(summary_file, "w") as f: 
		json.dump(results, f, indent=2)
