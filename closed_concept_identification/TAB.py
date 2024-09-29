import json
import os
import sys
from collections import defaultdict

sys.path.append("..")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import similarity
from args import parser
from data_utils import (get_clip_prediction, get_cls_id_to_label,
                        get_concept_set, get_topk_acc)
from sentence_utils import get_basename
from utils import get_similarity_from_activations

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")

	transformer_model = SentenceTransformer(args.sentence_transformer)

	concepts = get_concept_set(args.concept_set_file)
	cls_id_to_label = get_cls_id_to_label(args.network_class_file)
	
	similarity_names = ["cos-similarity", "cos_similarity_cubed", "rank_reorder", "wpmi", "soft_wpmi"]
	similarity_fns = [similarity.cos_similarity, similarity.cos_similarity_cubed, similarity.rank_reorder, similarity.wpmi, similarity.soft_wpmi]

	target_layers = args.target_layers.split(",")
	results = defaultdict(lambda: defaultdict(dict))
	args.save_activation_dir = os.path.join(args.save_activation_dir, f'{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}')

	for target_layer in target_layers:
		target_save_name = f"{args.save_activation_dir}/target_{args.probing_dataset}_{args.target_name}_{target_layer}.pt"
		audio_save_name = f"{args.save_activation_dir}/audio_{args.probing_dataset}.pt"
		text_save_name = f"{args.save_activation_dir}/text_{args.concept_set_file.split('/')[-1].replace('.txt', '')}.pt"

		for similarity_fn, similarity_name in zip(similarity_fns, similarity_names):

			similarities, target_feats = get_similarity_from_activations(target_save_name,   
				audio_save_name, text_save_name, similarity_fn, device=args.device)
			
			if target_layer == "fc":
				print(similarity_name)
				num_of_neuron = len([value for value in cls_id_to_label.values() if value != None])
				print(f"Calculate accuracy on {num_of_neuron} neurons out of {len(cls_id_to_label)} neurons")
				concepts = get_concept_set(args.concept_set_file, clip_format=False)

				print(f"TAB Top 1 acc:{get_topk_acc(similarities, cls_id_to_label, concepts, k=1):.4f}")
				print(f"TAB Top 5 acc:{get_topk_acc(similarities, cls_id_to_label, concepts, k=5):.4f}")


				prediction, gt = get_clip_prediction(similarities, cls_id_to_label, concepts, K=args.K, final_layer=True)

				# pred = torch.argmax(similarities, dim=1)
				# pred = [concepts[int(p)] for p in pred]
				
				gt_embeds = transformer_model.encode([cls_id_to_label[i] for i in range(len(cls_id_to_label))])
				pred_embeds = transformer_model.encode(prediction)
				cos_sim_mpnet = np.sum(pred_embeds * gt_embeds, axis=1)

				cos_sim = float(np.mean(cos_sim_mpnet))
				print("TAB cos sim   :{:.4f}".format(cos_sim))

				for i in range(len(similarities)):
					key = f"{target_layer}#{i}"
					results[key][similarity_name] = {'gt': gt[i], f'prediction': prediction[i]}
			else:
				prediction, _ = get_clip_prediction(similarities, cls_id_to_label, concepts, K=args.K, final_layer=False)

				for i in range(len(similarities)):
					key = f"{target_layer}#{i}"
					results[key][similarity_name] = {'gt': None, f'prediction': prediction[i]}

	if os.path.exists(args.save_prediction_dir) == False:
		os.makedirs(args.save_prediction_dir)

	with open(os.path.join(args.save_prediction_dir, f"tab-{args.target_name}-top{args.K}.json"), "w") as f: 
		json.dump(results, f, indent=2)
	