import numpy as np
import torch
from datasets import Audio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def tokenization(example):
	data = [a["array"] for a in example["audio"]]
	return processor(data)

def collate_batch(batch):
	"""
	Take a list of samples from a Dataset and collate them into a batch.
	Returns:
		A dictionary of tensors
	"""
	input_values = pad_sequence(torch.tensor(np.array([example[0] for example in batch])), batch_first=True, padding_value=0)
	labels = torch.stack([torch.tensor(example[1], dtype=torch.long) for example in batch])
	filenames = [d[-1] for d in batch]
	return {'input_values': input_values, 'labels': labels, 'filenames': filenames}

class ESC50Dataset(Dataset):
	def __init__(self, dataset, processor=None, mode='train', val_fold=1):
		self.data = []
		self.sampling_rate = 16000

		if mode == 'train':
			dataset = dataset.filter(lambda x: x["fold"] != val_fold)
		else:
			dataset = dataset.filter(lambda x: x["fold"] == val_fold)

		dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
		dataset = dataset.map(tokenization, batched=True)

		for line in dataset['train']:
			wav = torch.tensor(line["audio"]["array"]).to(torch.float32)
			processed_audio = np.array(wav).astype(np.float32)
			if processor is not None:
				processed_audio = processor(wav, sampling_rate = self.sampling_rate, return_tensor="pt")
				processed_audio = np.squeeze(np.array(processed_audio["input_values"]))
			obj = [processed_audio, line["target"], line['filename']]
			self.data.append(obj)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
