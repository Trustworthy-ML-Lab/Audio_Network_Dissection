# https://blog.csdn.net/JasonJarvan/article/details/79955664

import json
from tqdm import tqdm

ref_json = 'results/all_acoustic_adj_beats_unfreeze.json'
in_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf.json'
out_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf_llmf.json'

with open(ref_json, 'r') as f:
    ref = json.load(f)
ref_dict = {}
for word in ref:
    description = word['response'].lower()
    ans = True if 'yes' in description else False
    if type(word['word']) == list:
        ref_dict[word['word'][0]] = ans
    elif type(word['word']) == str:
        ref_dict[word['word']] = ans

with open(in_json, 'r') as f:
    data = json.load(f)
# print('ref_dict: ', ref_dict.keys())
newData = {}
for k in tqdm(data.keys()):
    d = data[k]
    adjs = d['adj_after_rbf']
    valid_adjs = []
    for adj in adjs:
        adj = adj.strip('\'')
        if ref_dict[adj] == True:
            valid_adjs.append(adj)

    d['adj_after_rbf_llmf'] = valid_adjs
    newData.update({k: d})

newData = json.dumps(newData, indent=4)
with open(out_json, 'w') as f:
    f.write(newData)
