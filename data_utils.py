import json
import os
from collections import defaultdict

import clip
import torch
import torchaudio
from datasets import Dataset, load_dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import similarity
from BEATs.myBeatsModel import MyBeatsModel
from sentence_utils import clean_repeated_substring, get_basename


def get_similarity_from_activations(activation_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # clip_feats:  torch.Size([2000, 50])
        clip_feats = (image_features @ text_features.T).softmax(dim=-1)
    
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(activation_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_similarity_from_descriptions(target_layer, clip_model, neuron_ordered_activation, descriptions_and_filenames, concept_features, similarity_fn, device="cuda"):

    similarities, keys = [], []
    for key, value in neuron_ordered_activation.items():
        if key.split('#')[0] != target_layer:
            continue
        activation = torch.FloatTensor(value['highly_activation_values']).unsqueeze(1).float().to(device)
        descriptions = [descriptions_and_filenames[f.split("/")[-1]] for f in value['highly_filename']]

        with torch.no_grad():
            descriptions = clip.tokenize(descriptions, truncate=True).to(device)
            description_features = clip_model.encode_text(descriptions).float()
            concept_features /= concept_features.norm(dim=-1, keepdim=True)

            description_features /= description_features.norm(dim=-1, keepdim=True)
            clip_feats = (100.0 * description_features @ concept_features.T).softmax(dim=-1)
            similarities.append(similarity_fn(clip_feats.float(), activation.float()))
            keys.append(key)

    similarities = torch.stack(similarities, dim=0).squeeze(1)
    return similarities, keys


def get_target_feature(activation_save_name):
    
    target_feats = torch.load(activation_save_name, map_location='cpu')

    return target_feats

def mean(list):
	
    return sum(list) / len(list)


def save_discriminative_sample(save_discriminative_sample_dir, save_activation_dir, probing_dataset, concept_set_file, target_name, target_layers, save_num=5):
    
    pil_data = get_data(probing_dataset, get_audio=False).to_pandas()
    pil_data = pil_data.iloc[:, :]

    discriminative_samples = defaultdict(lambda: defaultdict(list))

    for target_layer in target_layers:
        print("target layer:", target_layer)
        activation_save_name = os.path.join(save_activation_dir, f"target_{probing_dataset}_{target_name}_{target_layer}.pt")

        target_feats = get_target_feature(activation_save_name)

        if target_layer == "fc":
            save_num = target_feats.shape[0]

        top_vals, top_ids = torch.topk(target_feats, largest=True, k=save_num, dim=0)
        bot_vals, bot_ids = torch.topk(target_feats, largest=False, k=save_num, dim=0)
        
        for neuron_id in range(target_feats.shape[1]):
            key = target_layer + "#" + str(neuron_id)
            
            for top_id, top_val in zip(top_ids[:, neuron_id], top_vals[:, neuron_id]):
                top_id = top_id.item()
                top_val = top_val.item()
                data = pil_data.iloc[top_id, :]
                if probing_dataset == "esc50":    
                    discriminative_samples[key]["highly_label"].append(data["category"])
                    discriminative_samples[key]["highly_filename"].append(data["filename"])
                elif probing_dataset == "urban8k":
                    discriminative_samples[key]["highly_label"].append(int(data["classID"]))
                    discriminative_samples[key]["highly_filename"].append(data["slice_file_name"])
                elif probing_dataset == "gtzan":
                    discriminative_samples[key]["highly_label"].append(int(data["genre"]))
                    discriminative_samples[key]["highly_filename"].append(data["file"])                        
                discriminative_samples[key]["highly_activation_values"].append(float(top_val))

            for bot_id, bot_val in zip(bot_ids[:, neuron_id], bot_vals[:, neuron_id]):
                bot_id = bot_id.item()
                bot_val = bot_val.item()

                data = pil_data.iloc[bot_id, :]
                
                if probing_dataset == "esc50":    
                    discriminative_samples[key]["lowly_label"].append(data["category"])
                    discriminative_samples[key]["lowly_filename"].append(data["filename"])
                elif probing_dataset == "urban8k":
                    discriminative_samples[key]["lowly_label"].append(int(data["classID"]))
                    discriminative_samples[key]["lowly_filename"].append(data["slice_file_name"])
                elif probing_dataset == "gtzan":
                    discriminative_samples[key]["lowly_label"].append(int(data["genre"]))
                    discriminative_samples[key]["lowly_filename"].append(data["file"])                    
                discriminative_samples[key]["lowly_activation_values"].append(float(bot_val))


    if not os.path.exists(save_discriminative_sample_dir):
        os.makedirs(save_discriminative_sample_dir)
    
    # XXX we don't need concept set to find highly/lowly activated samples
    with open(f"{save_discriminative_sample_dir}/{target_name}_{probing_dataset}_{get_basename(concept_set_file)}.json", "w") as f: 
        json.dump(discriminative_samples, f, indent=2)

def get_description_dataset(audio_description_dir, save_activation_dir, probing_dataset, concept_set_file, target_name, target_layers, network_class_file, prompt_template, discriminative_type, K=5):
    
    pil_data = get_data(probing_dataset, get_audio=False).to_pandas()
    pil_data = pil_data.iloc[:, :]

    descriptions = get_audio_description(audio_description_dir, probing_dataset)
    cls_labels = get_cls_label(network_class_file)

    descriptions = clean_repeated_substring(descriptions)    
    
    dataset = defaultdict(list)
    
    for target_layer in target_layers:
        target_save_name = f"{save_activation_dir}/target_{probing_dataset}_{target_name}_{target_layer}.pt" 
        audio_save_name = f"{save_activation_dir}/audio_{probing_dataset}.pt"
        text_save_name = f"{save_activation_dir}/text_{get_basename(concept_set_file)}.pt"   
        
        _, target_feats = get_similarity_from_activations(target_save_name, audio_save_name, text_save_name, similarity.soft_wpmi)
        target_feats = get_target_feature(target_save_name)

        top_vals, top_ids = torch.topk(target_feats, largest=True, k=K, dim=0)
        bot_vals, bot_ids = torch.topk(target_feats, largest=False, k=K, dim=0)
        
        # target_feats [num_of_samples, num_of_neurons]
        for neuron_id in range(target_feats.shape[1]):
            # highly activated samples
            if target_layer == "fc":
                dataset["label"].append(cls_labels[(int(neuron_id))])
            # No inherent label for middle-layer neurons
            else: 
                dataset["label"].append("None")
            dataset["target_layer"].append(target_layer)
            dataset["neuron_id"].append(neuron_id)

            if discriminative_type == "highly":
                extract_discriminative_sample(dataset, pil_data, descriptions, neuron_id, top_ids, top_vals, probing_dataset, prompt_template, discriminative_type)
            elif discriminative_type == "lowly":
                extract_discriminative_sample(dataset, pil_data, descriptions, neuron_id, bot_ids, bot_vals, probing_dataset, prompt_template, discriminative_type)
            else:
                assert discriminative_type in ["highly", "lowly"]

    dataset = dict(dataset)
    dataset = Dataset.from_dict(dataset)

    return dataset

def extract_discriminative_sample(dataset, pil_data, descriptions, neuron_id, ids, activation_values, probing_dataset, prompt_template, discriminative_type):
    
    sample_labels = []
    sample_activation_values = []
    sample_filenames = []
    sample_descriptions = []
    for id, val in zip(ids[:, neuron_id], activation_values[:, neuron_id]):
        id = id.item()
        val = val.item()
        data = pil_data.iloc[id, :]
        
        filename_key = ""
        # TODO urban8k
        if probing_dataset == "esc50":    
            sample_labels.append(data["category"])
            filename_key = "filename"
        elif probing_dataset == "gtzan":
            sample_labels.append(data["genre"])
            filename_key = "file"

        sample_activation_values.append(val)
        sample_filenames.append(data[filename_key])
        try:
            sample_descriptions.append(descriptions[data[filename_key]])
        except: 
            print(f"Warning: missing the description of file", data[filename_key])

    dataset[f"{discriminative_type}_activated_sample_labels"].append(sample_labels)
    dataset[f"{discriminative_type}_activated_sample_activation_values"].append(sample_activation_values)
    dataset[f"{discriminative_type}_activated_sample_filenames"].append(sample_filenames)
    dataset[f"{discriminative_type}_activated_sample_descriptions"].append(sample_descriptions)

    dataset["raw_text"].append(sample_descriptions)
    dataset["audio_labels"].append(sample_labels)
    dataset["text"].append(prompt_template.format("\n".join(sample_descriptions))) 

    return dataset

def get_concept_dataset(save_summary_dir, probing_dataset, concept_set_file, target_name, target_layers, network_class, prompt_template, prediction_type="highly", K=5):

    with open(network_class, "r") as f:
        data = f.readlines()
        data = [d.split("\t") for d in data]
        cls_labels = sorted(data, key=lambda x: int(x[1].replace("\n", "")))
        cls_labels = [cls[0] for cls in cls_labels]
    
    with open(concept_set_file) as f:
        concepts = f.readlines()
        concepts = [c.replace("\n", "") for c in concepts]

    dataset = defaultdict(list)      

    if prediction_type == "highly":
        highly_activation_summary_file = os.path.join(save_summary_dir, f'{target_name}_{probing_dataset}_{concept_set_file.split("/")[-1].split(".txt")[0]}_highly_{K}.json')
        with open(highly_activation_summary_file) as f:
            highly_activation_summary = json.load(f)
        for line in highly_activation_summary:
            if not line["target_layer"] in target_layers:
                continue
            dataset["target_layer"].append(line["target_layer"])
            dataset["neuron_id"].append(line["neuron_id"])
            dataset["neuron_label"].append(line["neuron_label"])
            text = "concept set: \n"
            text += ", ".join(concepts)
            text += "\n\n"
            text += line["summary"]
            dataset["text"].append(prompt_template.format(", ".join(concepts), line["summary"]))

    elif prediction_type == "calibration":
        with open(os.path.join(save_summary_dir, f"summaries/split/calibration_{target_name}_esc50_esc50_5.json")) as f: 
            activation_summary = json.load(f)
        for ids, object in activation_summary.items():
            ids = ids.split("#")
            target_layer = ids[0]
            neuron_id = int(ids[1])
            if not target_layer in target_layers:
                continue
            dataset["target_layer"].append(target_layer)
            dataset["neuron_id"].append(neuron_id)
            dataset["neuron_label"].append(object["neuron_label"])
            text = ""
            # TODO !!
            for j, concept in enumerate(concepts):
                text += f"{concept} \n"

            dataset["text1"].append(text)
            dataset["text2"].append([" ".join(object["highly"])])
            dataset["text"].append(prompt_template.format(text, " ".join(object["highly"])))

    dataset = Dataset.from_dict(dict(dataset))
    return dataset


def get_target_model(target_name, device):
    if "ast-esc50" in target_name:
        target_model = AutoModelForAudioClassification.from_pretrained("Evan-Lin/ast-esc50").to(device)
    elif "ast-urban8k" in target_name:
        target_model = AutoModelForAudioClassification.from_pretrained("Evan-Lin/ast-urban8k").to(device)
    elif "ast-gtzan" in target_name:
        target_model = AutoModelForAudioClassification.from_pretrained("Evan-Lin/ast-gtzan").to(device)
    elif "beats-esc50-frozen" == target_name: 
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=50).to(device)
    elif "beats-esc50-finetuned" == target_name:
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=50).to(device)
    elif "beats-urban8k-frozen" == target_name: 
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=10).to(device)
    elif "beats-urban8k-finetuned" == target_name:
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=10).to(device)
    elif "beats-gtzan-frozen" == target_name: 
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=10).to(device)
    elif "beats-gtzan-finetuned" == target_name:
        target_model = MyBeatsModel(checkpoint_path="TODO", num_class=10).to(device)    
    else: 
        raise ValueError('Currently no this target model support')
    
    target_model.eval()
    return target_model

def get_data(dataset_name, get_audio):
         
    if dataset_name == "esc50":
        if get_audio:
            processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", sampling_rate = 16000)
        
            def handler_input(data):
                wav = torch.tensor(data["audio"]["array"]).to(torch.float32)
                resample_audio_1 = torchaudio.transforms.Resample(44100, 16000)(wav)
                resample_audio_2 = torchaudio.transforms.Resample(44100, 48000)(wav)
                data["input_values"] = processor(resample_audio_1, sampling_rate = 16000)["input_values"]
                data["raw_audio"] = resample_audio_2
                return data
            data = load_dataset("ashraq/esc50", keep_in_memory=False)["train"]
            data = data.map(handler_input, remove_columns=["audio"], batched=False)
        else: 
            data = load_dataset("ashraq/esc50", keep_in_memory=False)["train"]

    elif dataset_name == "urban8k":
        if get_audio:
            processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", sampling_rate = 16000)
        
            def handler_input(data):
                wav = torch.tensor(data["audio"]["array"]).to(torch.float32)
                resample_audio_1 = torchaudio.transforms.Resample(16000, 16000)(wav)
                resample_audio_2 = torchaudio.transforms.Resample(16000, 48000)(wav)
                data["input_values"] = processor(resample_audio_1, sampling_rate = 16000)["input_values"]
                data["raw_audio"] = resample_audio_2
                return data
            data = load_dataset("danavery/urbansound8K", keep_in_memory=False)["train"]
            data = data.filter(lambda x: int(x["fold"]) != 1)
            data = data.map(handler_input, remove_columns=["audio"], batched=False)
        else: 
            data = load_dataset("danavery/urbansound8K", keep_in_memory=False)["train"]   

    elif dataset_name == "gtzan":
        if get_audio:
            processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", sampling_rate = 16000)
        
            def handler_input(data):
                wav = torch.tensor(data["audio"]["array"]).to(torch.float32)
                resample_audio_1 = torchaudio.transforms.Resample(22050, 16000)(wav)
                resample_audio_2 = torchaudio.transforms.Resample(22050, 48000)(wav)
                data["input_values"] = processor(resample_audio_1, sampling_rate = 16000)["input_values"]
                data["raw_audio"] = resample_audio_2
                return data
            data = load_dataset("marsyas/gtzan", keep_in_memory=False)["train"]
            data = data.map(handler_input, remove_columns=["audio"], batched=False)
        else: 
            data = load_dataset("marsyas/gtzan", keep_in_memory=False)["train"]

        data =  Dataset.from_dict(data)  

    return data

def read_json(json_file):
    with open(json_file) as f: 
        return json.load(f)

def get_concept_set(concept_set_file, clip_format = False):

    with open(concept_set_file, 'r') as f: 
        concepts = (f.read()).split('\n') 
        concepts = [word.lower() for word in concepts]   
        if clip_format:
            concepts = [[word] for word in concepts]
            
    return concepts 

def get_label_to_cls(network_class_file):

    label_to_cls = {}
    label_to_cls[-1] = None
    with open(network_class_file) as f:
        all = f.readlines() 
        all = [line.replace("\n", "").split("\t") for line in all]
        for line in all: 
            label_to_cls[int(line[1])] = line[0]

    return label_to_cls

def get_cls_id_to_label(network_class_file):
    cls_id_to_label = {}
    with open(network_class_file) as f:	
        cls_name = f.read().split('\n')
        for cls in cls_name:
            cls_name, cls_id = tuple(cls.split("\t"))
            cls_id = int(cls_id)
            cls_id_to_label[cls_id] = cls_name
    
    return cls_id_to_label

def get_topk_acc(similarities, cls_id_to_label, concepts, k):
    total, correct = 0, 0
    for orig_id in range(len(cls_id_to_label)): # for each last layer neuron
        if cls_id_to_label[orig_id] == None:
            print("Warning: There is a last neuron without label name")
            continue
        else:
            vals, ids = torch.topk(similarities[orig_id], k, largest=True)
            ids = ids.tolist()
            # top-K prediction
            if cls_id_to_label[orig_id] in [concepts[i] for i in ids[:k]]:
                correct += 1
            total += 1

    return (correct / total) * 100	if total != 0 else 0

def get_clip_prediction(similarities, cls_id_to_label, concepts, K=1, final_layer=False):
    predictions, gt = [], []
    for orig_id in range(len(similarities)):
        vals, ids = torch.topk(similarities[orig_id], K, largest=True)
        pred = []
        for idx in ids: # top-K results
            pred.append(concepts[idx])
        
        predictions.append(pred)
        if final_layer:
            gt.append(cls_id_to_label[orig_id])
    
    return predictions, gt

def get_audio_description(audio_description_dir, probing_dataset, clip_format=False):
    
    file = os.path.join(audio_description_dir, f"salmon_{probing_dataset}.json")
    with open(file) as f: 
        descriptions = json.load(f)    

    if clip_format:
        descriptions = [[des] for des in descriptions.values()]
        
    return descriptions

def get_discriminative_sample(save_discriminative_sample_dir, target_name, concept_set_file, probing_dataset, K):

    file = os.path.join(save_discriminative_sample_dir, f"{target_name}_{concept_set_file.split('/')[-1]}_{probing_dataset}_{K}.json")
    with open(file) as f: 
        discriminative_samples =  json.load(f)    
    
    return discriminative_samples

def get_clustering(file):
    with open(file) as f: 
        clustering = json.load(f)
    
    return clustering

def get_concept_id_to_cls_label(concept_set_file, cls_class_file):
    with open(cls_class_file, "r") as f: 
        cls_class = f.read().split("\n")

    with open(concept_set_file, "r") as f: 
        concept_set = f.read().split("\n")
        concept_set = [c.lower() for c in concept_set]

    id_to_cls_label = {}
    for cls in cls_class:
        name = cls.split("\t")[0].lower()
        found = (name in concept_set)
        cls_id = int(cls.split("\t")[1])

        if found: 
            id_to_cls_label[cls_id] = name 
        else:
            id_to_cls_label[cls_id] = None 
    
    return id_to_cls_label

def get_cls_label(network_class_file):

    with open(network_class_file) as f:
        data = f.readlines()
        data = [d.split("\t") for d in data]
        cls_labels = sorted(data, key=lambda x: int(x[1].replace("\n", "")))
        cls_labels = [cls[0] for cls in cls_labels]

    return cls_labels
