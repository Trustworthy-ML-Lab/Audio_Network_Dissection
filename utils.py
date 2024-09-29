import json
import math
import os
import re
from collections import defaultdict

import clip
import laion_clap
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import similarity
from data_utils import (get_concept_id_to_cls_label, get_concept_set, get_data,
                        get_target_model)

PM_SUFFIX = {"max":"_max", "avg":""}

def get_activation(outputs, mode, layer="fc"):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode == 'avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            # TODO
            elif len(output.shape)==3: 
                if layer == "fc":
                    outputs.append(output[:, 0].clone())
                else: 
                    outputs.append(output[0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())

    # elif mode=='max':
    #     def hook(model, input, output):
    #         if len(output.shape)==4: #CNN layers
    #             outputs.append(output.amax(dim=[2,3]).detach())
    #         elif len(output.shape)==3: #ViT
    #             outputs.append(output[:, 0].clone())
    #         elif len(output.shape)==2: #FC layers
    #             outputs.append(output.detach())
    return hook

def collate_fn(batch):
    
    input_values = torch.stack([torch.tensor(b["input_values"]) for b in batch])

    return input_values

def collate_raw_fn(batch):

    input_values = pad_sequence([torch.tensor(b["raw_audio"]) for b in batch], batch_first=True)
    
    return input_values

def categorize_true_and_wrong_neuron(similarities, concepts, id_to_label):
	correct_id = []
	wrong_id = []
	skip_id = []
	for orig_id in range(len(similarities)):
		#skip classes not in audioset
		if id_to_label[orig_id] == None:
			skip_id.append(orig_id)
			continue
		else:
			vals, ids = torch.topk(similarities[orig_id], 1, largest=True)
			flag = False
			for idx in ids[:1]:
				if ((concepts[idx])==(id_to_label[orig_id])):
					correct_id.append(orig_id)
					flag = True
					break
				else:
					print((concepts[idx]) , (id_to_label[orig_id]))
			if flag == False: 
				wrong_id.append(orig_id)
			
	return correct_id, wrong_id, skip_id

def save_target_activations(target_model, target_name, dataset, save_name, target_layers=["layer4"], batch_size=4, device="cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    target_layers = target_layers.split(",")
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return

    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    print(target_model)
    for target_layer in target_layers:
        print("Command is: ", target_layer)
        command = ""
        # command = "target_model.{}.register_forward_hook(get_activation(all_features['fc'], pool_mode))".format('classifier.dense')

        if "ast" in target_name:
            if target_layer == "fc":
                command = "target_model.{}.register_forward_hook(get_activation(all_features['fc'], pool_mode))".format('classifier.dense')
            elif "layer" in target_layer and "attention_output" in target_layer:
                number = int(re.findall(r'\d+', target_layer)[0])
                command = "target_model.audio_spectrogram_transformer.encoder.layer[{}].attention.output.dense.register_forward_hook(get_activation(all_features['{}'], pool_mode))".format(number, target_layer)
            elif "layer" in target_layer and "output" in target_layer:
                number = int(re.findall(r'\d+', target_layer)[0])
                command = "target_model.audio_spectrogram_transformer.encoder.layer[{}].output.dense.register_forward_hook(get_activation(all_features['{}'], pool_mode))".format(number, target_layer)
            elif "layer" in target_layer and "intermediate" in target_layer:
                number = int(re.findall(r'\d+', target_layer)[0])
                command = "target_model.audio_spectrogram_transformer.encoder.layer[{}].intermediate.dense.register_forward_hook(get_activation(all_features['{}'], pool_mode))".format(number, target_layer)
        elif "beats" in target_name:
            if target_layer == "fc":
                command = "target_model.{}.register_forward_hook(get_activation(all_features['fc'], pool_mode))".format('fc')
            elif "layer" in target_layer:
                name = target_layer.split("_")
                fc_number = int(name[1]) 
                encoder_number = int(re.findall(r'\d+',name[0])[0])
                command = "target_model.beats.encoder.layers[{}].fc{}.register_forward_hook(get_activation(all_features['{}'], pool_mode, layer='not_fc'))".format(encoder_number, fc_number, target_layer)


        print(command)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        if "beats" in target_name:
            for audios in tqdm(DataLoader(dataset, batch_size, num_workers=0, pin_memory=True, collate_fn=collate_raw_fn)):
                # print("audio_shape", audios.shape)
                audios = torch.tensor(audios)
                _ = target_model(audios.to(device))
        elif "ast" in target_name:
            for audios in tqdm(DataLoader(dataset, batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn)):
                audios = torch.tensor(audios.squeeze(1))
                # [batch_size, 1024, 128]
                # print("audio_shape", audios.shape)
                _ = target_model(audios.to(device))    
                # break
    
    # all_features[target_layer] size [step numbers, batch_size]
    for target_layer in target_layers:    
        print("Saved target_layer: ", target_layer)
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()

    del all_features
    torch.cuda.empty_cache()
    return

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def save_clap_activations(model, probing_dataset, texts, audio_save_name, text_save_name, batch_size):
    if not os.path.exists(audio_save_name):
        _make_save_dir(audio_save_name)
        audio_features = []
        with torch.no_grad():
            for audios in tqdm(DataLoader(probing_dataset, batch_size, num_workers=0, pin_memory=True, collate_fn=collate_raw_fn)):
                audios = np.array(audios)
                audios = torch.from_numpy(int16_to_float32(float32_to_int16(audios))).float()
                audio_embed = model.get_audio_embedding_from_data(audios, use_tensor=True)
                audio_features.append(audio_embed)
        audio_features = torch.cat(audio_features, dim=0)
        torch.save(audio_features, audio_save_name)
        del audio_features
        torch.cuda.empty_cache()
    else: 
        print(f"file exist, load {text_save_name}")
    if not os.path.exists(text_save_name):
        _make_save_dir(text_save_name)
        text_features = []
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(texts) / batch_size))):
                text = [t[0] for t in texts[batch_size*i:batch_size*(i+1)]]
                text_embed = model.get_text_embedding(text)
                text_features.append(torch.tensor(text_embed))
        text_features = torch.cat(text_features, dim=0)
        torch.save(text_features, text_save_name)
        del text_features
        torch.cuda.empty_cache()
    else: 
        print(f"file exist, load {text_save_name}")
    return

def save_activations(args):
    
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt() # download the default pretrained checkpoint. 

    clap_model = clap_model.to(args.device)

    target_model = get_target_model(args.target_name, args.device)

    probing_data = get_data(args.probing_dataset, get_audio=True)

    # load concept set
    with open(args.concept_set_file, 'r') as f: 
        concept = (f.read()).split('\n')
        # ignore empty lines
        concepts = [[i] for i in concept if i != ''] 
    # concepts = get_concept_set(args.concept_set_file)

    save_activation_dir = args.save_activation_dir

    # XXX format string
    target_layer = '{}'
    target_save_name = f"{save_activation_dir}/target_{args.probing_dataset}_{args.target_name}_{target_layer}{PM_SUFFIX[args.pool_mode]}.pt"
    audio_save_name = f"{save_activation_dir}/audio_{args.probing_dataset}.pt"
    text_save_name = f"{save_activation_dir}/text_{args.concept_set_file.split('/')[-1].replace('.txt', '')}.pt"

    save_clap_activations(clap_model, probing_data, concepts, audio_save_name, text_save_name, args.batch_size)
    
    save_target_activations(target_model=target_model, target_name=args.target_name, dataset=probing_data, save_name=target_save_name, target_layers=args.target_layers, batch_size=args.batch_size, device=args.device, pool_mode=args.pool_mode)

    return

def class_prediction(probing_dataset, save_activation_dir, save_description_dir,  concept_set, network_class, target_name, target_layers, K=1, device="cuda"):
    # clip_model = laion_clap.CLAP_Module(enable_fusion=False)
    # clip_model.load_ckpt() # download the default pretrained checkpoint. 

    # clip_model = clip_model.to(device)
    results = defaultdict(list)
    model, _ = clip.load("ViT-B/32", device=device)

    pil_data = get_data(probing_dataset, get_audio=False).to_pandas()
    pil_data = pil_data.iloc[:, :]

    with open(network_class, 'r') as f:	
        cls_name = f.read().split('\n')
        cls_id_to_name = [-1 for _ in cls_name]
        for cls in cls_name:
            cls_name, cls_id = tuple(cls.split("\t"))
            cls_id = int(cls_id)
            cls_id_to_name[cls_id] = cls_name

    with torch.no_grad():
        description_file = f"{save_description_dir}/salmon_{probing_dataset}.json"
        with open(description_file) as f:
            all_description = json.load(f)
        all_description_list = [d for d in all_description.values()]
        print(len(all_description_list))
        text = clip.tokenize(all_description_list, truncate=True).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if type(target_layers) == str:
            target_layers = target_layers.split(",")

        # TODO
        # concept_set = "fine-grained_concept.txt"
        with open(concept_set, 'r') as f: 
            concepts = (f.read()).split('\n') 
            concepts = [word.lower() for word in concepts]
            word = clip.tokenize(concepts, truncate=True).to(device)
            word_features = model.encode_text(word)
            word_features /= word_features.norm(dim=-1, keepdim=True)

        print(word_features.shape)
        print(text_features.shape)

        clip_feats = (100.0 * text_features @ word_features.T).softmax(dim=-1) # (10, 50)


        for target_layer in target_layers:
            print("target layer:", target_layer)

            target_save_name = f"{save_activation_dir}/target_{probing_dataset}_{target_name}_{target_layer}.pt" 
            target_feats = torch.load(target_save_name, map_location='cpu')

            # top_vals, top_ids = torch.topk(target_feats, largest=True, k=K, dim=0)
            # top_vals, top_ids = torch.sort(target_feats, descending=True, dim=0)
            # bot_vals, bot_ids = torch.topk(target_feats, largest=False, k=K, dim=0)
            # [audio numbers, neuron numbers]
            print("target size", target_feats.shape)

            for neuron_id in range(target_feats.shape[1]):
                key = target_layer + "#" + str(neuron_id)
                activation = target_feats[:, neuron_id].unsqueeze(1)
                similarities = similarity.cos_similarity_cubed(clip_feats.float(), activation.float())
                similarities = similarities.squeeze(0)
                # cls_idx = torch.argmax(similarities)
                _, ids = torch.topk(similarities, K, largest=True)
                ids = ids.tolist()
                for cls_id in ids:
                    # print(cls_id_to_name[cls_id])
                    # results[key].append(concepts[cls_id])
                    results[key].append(cls_id_to_name[cls_id])
                # [concept_num (network class nun)]
                # print("sim", similarities.shape)

    with open(f"predict_class/{target_name}-{K}.json", "w") as f:
        json.dump(results, f, indent=2)

def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T) 
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=4):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    # clip
    pred_tokens = [[p] for p in preds]
    gt_tokens = [[g] for g in gt]

    # pred_tokens = clip.tokenize(preds).to(device)
    # gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            # clip model
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))
            
        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return
