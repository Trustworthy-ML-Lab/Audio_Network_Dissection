import json
import os
import sys
from collections import defaultdict

sys.path.append("..")

import clip
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import similarity
from args import parser
from data_utils import (get_audio_description, get_clip_prediction,
                        get_cls_id_to_label, get_concept_set,
                        get_similarity_from_descriptions, get_topk_acc,
                        read_json)
from sentence_utils import get_basename

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")

	clip_model, clip_preprocess = clip.load(args.clip_model, device=args.device)
	transformer_model = SentenceTransformer(args.sentence_transformer)

	concepts = get_concept_set(args.concept_set_file)
	descriptions = get_audio_description(args.audio_description_dir, args.probing_dataset)

	cls_id_to_label = get_cls_id_to_label(args.network_class_file)

	discriminative_sample_file = os.path.join(args.save_discriminative_sample_dir, f"{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}.json")
	neuron_ordered_activation = read_json(discriminative_sample_file)

	# similarity_names = ["cos_similarity_cubed"]
	# similarity_fns = [similarity.cos_similarity_cubed]
	similarity_names = ["cos-similarity", "cos_similarity_cubed", "rank_reorder", "wpmi", "soft_wpmi"]
	similarity_fns = [similarity.cos_similarity, similarity.cos_similarity_cubed, similarity.rank_reorder, similarity.wpmi, similarity.soft_wpmi]

	target_layers = args.target_layers.split(",")
	results = defaultdict(lambda: defaultdict(dict))
	args.save_activation_dir = os.path.join(args.save_activation_dir, f'{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}')

	with open(args.concept_set_file, 'r') as f: 
		concept_features = []
		bs = 100
		for step in range(0, len(concepts) // bs + 1):
			con = concepts[bs*step: bs*(step+1)]
			word = clip.tokenize(con, truncate=True).to(args.device)
			word_feature = clip_model.encode_text(word).detach().cpu()
			concept_features.append(word_feature)
		concept_features = torch.cat(concept_features, 0).to(args.device).float()

	for target_layer in target_layers:
		for similarity_fn, similarity_name in zip(similarity_fns, similarity_names):
			
			similarities, keys = get_similarity_from_descriptions(target_layer, clip_model, 
				neuron_ordered_activation, descriptions, concept_features, similarity_fn, device=args.device)

			print('similarity function: ',similarity_name)
			if target_layer == 'fc':
				num_of_neuron = len([value for value in cls_id_to_label.values() if value != None])
				print(f"Calculate accuracy on {num_of_neuron} neurons out of {len(cls_id_to_label)} neurons")

				print(f"{similarity_name} Top-1 fc acc:{get_topk_acc(similarities, cls_id_to_label, concepts, k=1):.4f}")
				print(f"{similarity_name} Top-5 fc acc:{get_topk_acc(similarities, cls_id_to_label, concepts, k=5):.4f}")

				pred = [concepts[int(p)] for p in torch.argmax(similarities, dim=1)]

				gt_embeds = transformer_model.encode([cls_id_to_label[i] for i in range(len(cls_id_to_label))])
				pred_embeds = transformer_model.encode(pred)
				cos_sim_mpnet = np.sum(pred_embeds * gt_embeds, axis=1)

				cos_sim = float(np.mean(cos_sim_mpnet))
				print("DB {} fc cos sim: {:.4f}".format(similarity_name, cos_sim))

			prediction, gt = get_clip_prediction(similarities, cls_id_to_label, concepts, K=args.K, final_layer=(target_layer=='fc'))

			for i in range(len(similarities)):
				results[keys[i]][similarity_name] = {'gt': gt[i], f'prediction': prediction[i]} if target_layer == 'fc' else {'gt': None, 'prediction': prediction[i]}

	if not os.path.exists(args.save_prediction_dir):
		os.makedirs(args.save_prediction_dir)

	with open(os.path.join(args.save_prediction_dir, f"db-{args.target_name}-top{args.K}.json"), "w") as f: 
		json.dump(results, f, indent=2)
