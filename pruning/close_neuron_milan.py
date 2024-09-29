### currently only support AST
### before running this code, please make sure you have run fig3_adjective_distribution.ipynb to process the output of open-concept identification module.
import os
os.chdir('..')
import numpy as np
import json
import pickle
import torch
from tqdm import tqdm
import random
from transformers import AutoProcessor, AutoModelForAudioClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter
from datasets import Audio, load_dataset
import torch.nn.utils.prune as prune
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt


processor = AutoProcessor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')

def tokenization(example):
	data = [a['array'] for a in example['audio']]
	return processor(data)

class AudioDataset(Dataset):
	def __init__(self, dataset, processor, mode='train', val_fold=1):
		self.data = []
		self.sampling_rate = 16000

		if mode == 'train':
			dataset = dataset.filter(lambda x: x['fold'] != val_fold)
		else:
			dataset = dataset.filter(lambda x: x['fold'] == val_fold)

		dataset = dataset.cast_column('audio', Audio(sampling_rate=self.sampling_rate))
		dataset = dataset.map(tokenization, batched=True)

		for line in dataset['train']:
			wav = torch.tensor(line['audio']['array']).to(torch.float32)
			processed_audio = processor(wav, sampling_rate = self.sampling_rate, return_tensor='pt')
			processed_audio = np.squeeze(np.array(processed_audio['input_values']))
			obj = [processed_audio, line['target'], line['filename']]
			self.data.append(obj)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def collate_batch(batch):
	'''
	Take a list of samples from a Dataset and collate them into a batch.
	Returns:
		A dictionary of tensors
	'''
	input_values = pad_sequence(torch.tensor(np.array([example[0] for example in batch])), batch_first=True, padding_value=0)
	labels = torch.stack([torch.tensor(example[1], dtype=torch.long) for example in batch])
	filenames = [d[-1] for d in batch]
	return {'input_values': input_values, 'labels': labels, 'filenames': filenames}


def compar_func(x, task):
	if task in ['preps', 'verbs', 'nouns', 'caption_len', 'adj_after_rbf_llmf']:
		y = x[task]
	elif task == 'basic_adj':
		y = set(x['adj_after_rbf_llmf']).intersection(set(basic_adjs))
	elif task == 'high_adj':
		y = set(x['adj_after_rbf_llmf']).difference(set(basic_adjs))
	elif task in ['clear', 'high-pitched', 'high', 'loud']:
		y = set(x['adj_after_rbf_llmf']).intersection(set([task]))
	else:
		raise Exception('task not implemented')
	if type(y) == int:
		return y
	return len(y)


network_class_file = 'data/network_class/esc50.txt'
pos = ['random', 'preps', 'verbs', 'nouns', 'caption_len', 'adj_after_rbf_llmf'] # exp1
highlowadj = ['random', 'basic_adj', 'high_adj'] # exp2
basic_adjs = ['random', 'clear', 'high-pitched', 'high', 'loud'] # exp3. 'high' is 'high-quality'.

seeds = [20, 202, 2024]
exp = 'fig4a'
if exp == 'fig4a':
	tasks = pos # the only parameter
elif exp == 'fig4b':
    tasks = highlowadj
elif exp == 'fig4c':
    tasks = basic_adjs
else:
    raise Exception('exp not implemented')

input_dimension = {'attention_output': 768, 'intermediate': 768, 'output': 3072}
label2class = {-1: None}
with open(network_class_file) as f:
	label_neuron_match = f.readlines() 
	label_neuron_match = [line.replace('\n', '').split('\t') for line in label_neuron_match]
	for line in label_neuron_match:
		label2class[int(line[1])] = line[0]

in_json = '/work/yuxiang1234/CLAP-dissect/summaries/split/calibration_ast-esc50_esc50_esc50_5_adj_rbf_llmf_allPOS_maxWordDiff.json'
with open(in_json, 'r') as f:
	data = json.load(f)
	ori_data = list(data.items())


max_pruned_nums = [6442, 5535, 4428, 3321, 2214, 1107] # 12% ~ 2% of all the 55346 neurons in AST
processor = AutoProcessor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
dataset = load_dataset('ashraq/esc50')
if (os.path.exists('dev_dataset.pickle')):
	print('cache exist, load cache')
	with open('dev_dataset.pickle', 'rb') as f:
		dev_dataset = pickle.load(f)
else:
	print('cache not exist')
	dev_dataset = AudioDataset(dataset, processor, mode='dev')
	with open('dev_dataset.pickle', 'wb') as f:
		pickle.dump(dev_dataset, f)

results = {}
for task in tqdm(tasks):
	results[task] = {}
	for max_pruned_num in max_pruned_nums:
		results[task][max_pruned_num] = defaultdict(list)
		for seed in seeds:
			data = deepcopy(ori_data)
			random.Random(seed).shuffle(data)
			data = dict(data)

			if task == 'random' or task in basic_adjs:
				sorted_data = data
			else:
				for k in data.keys():
					data[k].update({'att': compar_func(data[k], task)})
					
				sorted_data = sorted(data.items(), key=lambda x: x[1]['att'], reverse=True)
				sorted_data = dict(sorted_data)

			dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn=collate_batch)
			mask_cnt = 0
			masked_neuron = defaultdict(list)
			masked_neuron_bias = defaultdict(list)
			model = AutoModelForAudioClassification.from_pretrained('Evan-Lin/ast-esc50', num_labels=50, ignore_mismatched_sizes=False)
			model = model.to('cuda')
			for key, item in tqdm(sorted_data.items()):
				layer = key.split('#')[0]
				neuron_id = key.split('#')[1]

				if layer == 'fc':
					continue

				layer = layer.split('_')
				layer_num, layer_name = layer[0], layer[1]
				if len(layer) == 3:
					layer_name = layer[1] + '_' + layer[2]
				
				layer_id = layer_num.strip('layer')
				dim = input_dimension[layer_name.strip(str(layer_id))]
				neuron_adjs = item['adj_after_rbf_llmf']
	
				if (task in pos + highlowadj or task == 'random' or (task in basic_adjs and item['att'] > 0)) and mask_cnt < max_pruned_num:
					mask_cnt += 1
					masked_neuron[layer_num + '_' + layer_name].append([0 for _ in range(dim)])
					masked_neuron_bias[layer_num + '_' + layer_name].append(0)
				else:
					masked_neuron[layer_num + '_' + layer_name].append([1 for _ in range(dim)])
					masked_neuron_bias[layer_num + '_' + layer_name].append(1)
			
			for key, mask in masked_neuron.items():
				layer_id = int(key.split('_')[0].strip('layer'))
				layer_name = key.split('_')[1]
				if layer_name == 'attention':
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].attention.output.dense
				elif layer_name == 'intermediate':
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].intermediate.dense
				elif layer_name == 'output':
					module = model.audio_spectrogram_transformer.encoder.layer[layer_id].output.dense
				weight_mask = torch.tensor(mask).to('cuda')
				bias_mask = torch.tensor(masked_neuron_bias[key]).to('cuda')

				prune.custom_from_mask(module, 'weight', mask=weight_mask)
				prune.custom_from_mask(module, 'bias', mask=bias_mask)
			
			correct, total = 0, 0
			with torch.no_grad():
				for batch in tqdm(dev_loader):
					batch['input_values'] = batch['input_values'].to('cuda')
					batch['labels'] = batch['labels'].to('cuda')
					outputs = model(batch['input_values'])
					outputs = outputs.logits
					outputs_list = outputs.detach().cpu().tolist()
					outputs = torch.argmax(outputs, dim = -1)
					labels = batch['labels']
					correct += torch.sum(outputs == labels).detach().cpu().item()
					total += outputs.shape[0]

			results[task][max_pruned_num]['correct'].append(correct)
			results[task][max_pruned_num]['total'].append(total)

		assert all(x == results[task][max_pruned_num]['total'][0] for x in results[task][max_pruned_num]['total']), print('not all the same')
		avg_acc = sum(results[task][max_pruned_num]['correct']) / len(results[task][max_pruned_num]['correct'])
		print(f'{task}_{max_pruned_num} performance: ', avg_acc / results[task][max_pruned_num]['total'][0])
		print(f'{task}_{max_pruned_num} performance drop: ', 0.95 - avg_acc / results[task][max_pruned_num]['total'][0])
		results[task][max_pruned_num] = avg_acc / results[task][max_pruned_num]['total'][0]

print('results: ', results)
x = list(range(0, 14, 2))
markers = ['^', 'o', 's', '*', 'x', 'D']
plt.rcParams.update({'font.size': 13})
for i, task in enumerate(tasks):
	plt.plot(x, results[task], label=task, marker=markers[i])

plt.ylabel('classification accuracy', fontsize=20)
plt.xlabel('% of neurons pruned', fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig(f'{exp}.pdf', format='pdf', bbox_inches='tight')
