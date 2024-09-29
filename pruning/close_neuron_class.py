import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.utils.prune as prune
from audio_dataset import ESC50Dataset, collate_batch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForAudioClassification, AutoProcessor

sys.path.append("../")
from args import parser
from data_utils import (get_cls_label, get_label_to_cls, get_target_model,
                        read_json)
from sentence_utils import get_basename


def main():
	
	args = parser.parse_args()
	os.chdir("../")
		
	random.seed(args.seed)

	label_to_cls = get_label_to_cls(args.network_class_file)

	if args.pruning_strategy == "tab":
		prediction_file = os.path.join(args.save_prediction_dir, f"tab-{args.target_name}-top{args.K}.json")
	elif args.pruning_strategy == "db": 
		prediction_file = os.path.join(args.save_prediction_dir, f"db-{args.target_name}-top{args.K}.json")
	# "random" needs neurons names
	elif args.pruning_strategy == "ocp" or args.pruning_strategy == "random": 
		prediction_file = os.path.join(args.save_summary_dir, f"calibration_{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}_top{args.K}.json")

	prediction = read_json(prediction_file)
	if args.pruning_strategy == "random":
		_ = list(prediction.items())
		random.Random(args.seed).shuffle(_)
		prediction = dict(_)

	if "ast" in args.target_name:
		input_dimension = {"attention_output":768, "intermediate": 768, "output":3072}
		processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")	
	elif "beats" in args.target_name: 
		input_dimension = {"1": 768, "2": 3072}
		processor = None

	dataset = load_dataset("ashraq/esc50")
	if (os.path.exists(f"{args.target_name}_dev_dataset.pickle")):
		print("cache exist, load cache")
		with open(f"{args.target_name}_dev_dataset.pickle", "rb") as f:
			dev_dataset = pickle.load(f)
	else:
		print("cache not exist")
		dev_dataset = ESC50Dataset(dataset, processor, mode='dev')
		with open(f"{args.target_name}_dev_dataset.pickle", "wb") as f:
			pickle.dump(dev_dataset, f)

	dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False, pin_memory=True, collate_fn=collate_batch)

	results = {}

	for cls_id, cls_name in label_to_cls.items():

		mask_cnt = 0 
		pruned_neuron_record = defaultdict(int)
		masked_out_neurons = []
		masked_neuron = defaultdict(list)
		masked_neuron_bias = defaultdict(list)
		
		model = get_target_model(args.target_name, device=args.device)

		for key, _ in tqdm(prediction.items()):
			layer = key.split('#')[0]

			if layer == 'fc':
				continue

			layer = layer.split('_')

			layer_name = layer[1]
			layer_num = layer[0]

			if len(layer) == 3:
				layer_name = layer[1] + '_' + layer[2]
			
			layer_id = layer_num.replace("layer", "")

			neuron_id = key.split('#')[1]
			dim = input_dimension[layer_name]

			flag = False
			if args.pruning_strategy == "ocp":
				nouns = prediction[key]['nouns']

				if cls_name is not None:
					for n in nouns:
						if n in cls_name: 
							flag = True 
							break		

			# We select the best simlarity function by 
			elif args.pruning_strategy == "tab" :
				if cls_name is not None and cls_name in prediction[key]["soft_wpmi"]["prediction"][:3]:
					flag = True
			elif args.pruning_strategy == "db":
				if cls_name is not None and cls_name in prediction[key]["cos_similarity_cubed"]["prediction"][:3]:
					flag = True
			elif args.pruning_strategy == "random":
				if mask_cnt < args.max_pruned_num:
					flag = True

			if flag:
				pruned_neuron_record[layer_num + '_' + layer_name] += 1
				mask_cnt += 1
				masked_neuron[layer_num + "_" + layer_name].append([0 for _ in range(dim)])
				masked_out_neurons.append(f"{layer_name}_{layer_id}#{neuron_id}")
				masked_neuron_bias[layer_num + "_" + layer_name].append(0)
			else:
				masked_neuron[layer_num + "_" + layer_name].append([1 for _ in range(dim)])
				masked_neuron_bias[layer_num + "_" + layer_name].append(1)

		for key, mask in masked_neuron.items():
			layer_id = key.split("_")[0].replace("layer", "")
			layer_id = int(layer_id)
			layer_name = key.split("_")[1]
			
			if "ast" in args.target_name:
				if layer_name == "attention":
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].attention.output.dense
				elif layer_name == "intermediate":
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].intermediate.dense
				elif layer_name == "output":
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].output.dense
			elif "beats" in args.target_name: 
				if layer_name == "1":
					module = model.beats.encoder.layers[layer_id].fc1
				elif layer_name == "2": 
					module = model.beats.encoder.layers[layer_id].fc2

			weight_mask = torch.tensor(mask).to("cuda")
			bias_mask = torch.tensor(masked_neuron_bias[key]).to("cuda")

			prune.custom_from_mask(module, 'weight', mask=weight_mask)
			prune.custom_from_mask(module, 'bias', mask=bias_mask)
		
		wrong_record = []
		correct_by_class, total_by_class = defaultdict(int), defaultdict(int)
		pred_by_class = defaultdict(int)
		correct, total = 0, 0
		confidence_by_class = defaultdict(list)
		
		with torch.no_grad():
			for batch in tqdm(dev_loader):
				batch["input_values"] = batch["input_values"].to("cuda")
				batch["labels"] = batch["labels"].to("cuda")
				outputs = model(batch["input_values"])
				if "ast" in args.target_name:
					outputs = outputs.logits
				outputs_list = outputs.detach().cpu().tolist()
				outputs = torch.argmax(outputs, dim = -1)
				labels = batch["labels"]
				correct += torch.sum(outputs == labels).detach().cpu().item()
				total += outputs.shape[0]	
				
				outputs = outputs.detach().cpu().tolist()
				labels = labels.detach().cpu().tolist()
				for idx, (pred, gt, filename) in enumerate(zip(outputs, labels, batch['filenames'])):
					pred = label_to_cls[pred]
					gt = label_to_cls[gt]
					if (pred == gt):
						correct_by_class[pred] += 1
					else:
						wrong_record.append(filename)
					pred_by_class[pred] += 1
					total_by_class[gt] += 1
					confidence_by_class[gt].append(outputs_list[idx])

		results[cls_name] = {}
		results[cls_name]["masked_count"] = len(masked_out_neurons)
		results[cls_name]["masked_neuron"] = masked_out_neurons
		results[cls_name]["correct"] = correct
		results[cls_name]["total"] = total
		results[cls_name]["correct_by_class"] = correct_by_class
		results[cls_name]["pred_by_class"] = pred_by_class
		results[cls_name]["total_by_class"] = total_by_class
		results[cls_name]["confidence"] = confidence_by_class
		print(cls_name, 'mask_cnt: ', mask_cnt)

	if not os.path.exists(args.save_pruning_dir):
		os.makedirs(args.save_pruning_dir)

	with open(os.path.join(args.save_pruning_dir, f"class-{args.target_name}-{args.pruning_strategy}.json"), "w") as f:
		json.dump(results, fp = f,indent=2)

if __name__ == '__main__':
    main()
