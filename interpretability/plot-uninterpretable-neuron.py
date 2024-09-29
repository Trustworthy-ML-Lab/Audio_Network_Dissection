import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.append("..")
from args import parser
from sentence_utils import get_basename

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")

	discriminative_sample_file = os.path.join(args.save_discriminative_sample_dir, f"{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}.json")
	
	clustering_file = os.path.join(args.save_interpretability_dir, f"{args.probing_dataset}_clustering.json")
	with open(discriminative_sample_file) as f: 
		discriminative_samples =  json.load(f)

	with open(clustering_file) as f: 
		clustering = json.load(f)
  
    # Some audio lost when generating captions
	skip_cnt = 0
	skip_audio_ids = ['177621-0-0-54.wav', '151359-1-3-0.wav', '160093-3-0-0.wav', '160092-3-0-0.wav', '151359-1-1-0.wav', '151359-1-2-0.wav', '170243-1-0-0.wav', '184805-0-0-58.wav', '155130-1-0-0.wav', '155129-1-1-0.wav', '118496-1-0-0.wav', '17307-1-0-0.wav', '162702-1-0-0.wav', '151359-1-0-0.wav', '118496-1-1-0.wav', '155129-1-0-0.wav']

	Ks = [args.K / 5 * i for i in range(3, 6)]
	for K in Ks:
		number_per_layer = defaultdict(float)
		for key in discriminative_samples.keys():
			audios = discriminative_samples[key]["highly_filename"]
			member_num = defaultdict(set)
			uninterpretable = True
		
			for audio in audios: 
				audio = audio.split("/")[-1]
				if audio in skip_audio_ids:
					skip_cnt += 1
					print(skip_cnt)
					break
				for cluster_id in clustering[audio].keys():
					member_num[cluster_id].add(audio)
			for num in member_num.values():
				if len(num) >= K:
					uninterpretable = False
					break
			if uninterpretable:
				number_per_layer[key.split("#")[0]] += 1

		plot_data = defaultdict(list)
		if "ast" in args.target_name:
			neuron_number = {"attention_output": 768, "intermediate": 3072, "output": 768}
			layer_names = ["attention_output", "intermediate" , "output"]
			for i in range(12):
				for layer_name in layer_names:
					plot_data[layer_name].append(number_per_layer[f"layer{i}_{layer_name}"] / neuron_number[layer_name] * 100)
		
		elif "beats" in args.target_name:
			neuron_number = {"1": 3072, "2": 768}
			layer_names = ["1", "2"]
			for i in range(12):
				for layer_name in layer_names:
					plot_data[layer_name].append(number_per_layer[f"layer{i}_{layer_name}"] / neuron_number[layer_name] * 100)		

		x = range(1, 13)	
		plt.figure(figsize=(12, 10))
		
		# adjusting layout
		if  "beats" in args.target_name:
			plt.plot(x, plot_data[layer_names[0]], label="first linear layer", marker='o')  # Line plot for list1
			plt.plot(x, plot_data[layer_names[1]], label="second linear layer", marker='^') 	
		else:
			plt.plot(x, plot_data[layer_names[0]], label=layer_names[0], marker='o')  # Line plot for list1
			plt.plot(x, plot_data[layer_names[1]], label=layer_names[1], marker='^')  # Line plot for list2
		
		if "ast" in args.target_name:
			plt.plot(x, plot_data[layer_names[2]], label=layer_names[2], marker='s')  # Line plot for list3

		# Adding titles and labels
		plt.xlabel('layers', fontsize=36)
		plt.ylabel('Percent (%)', fontsize=36)
		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.ylim((0, 100))
		plt.xlim((1, 13))
		if K == 3 or K == 4:
			plt.legend(fontsize=30, loc='upper left')
		else: 
			plt.legend(fontsize=30, loc='lower left')

		# Show the plot
		plt.subplots_adjust(top=0.98)
		plt.show()

		save_dir = f"interpretability/top-{args.K}_activated"
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		plt.savefig(os.path.join(save_dir, f"interpretability-{int(K)}-{args.target_name}.pdf"), format="pdf")			
