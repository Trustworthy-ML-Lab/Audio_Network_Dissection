# https://blog.csdn.net/JasonJarvan/article/details/79955664

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_lg')

# in_json = 'calibration_ast-esc50_esc50_esc50_5_adj_rbf_llmf_allPOS.json'
# out_json = 'calibration_ast-esc50_esc50_esc50_5_adj_rbf_llmf_allPOS_maxWordDiff.json'
in_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf_llmf_allPOS.json'
out_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf_llmf_allPOS_maxWordDiff.json'
with open(in_json, 'r') as f:
    data = json.load(f)
# print('stopwords.words: ', stopwords.words('english'))

newData = {}
for k in tqdm(data.keys()):
    d = data[k]
    sentence = ''
    for point in d['highly']:
        sentence = sentence + ' ' + point
    doc = nlp(sentence.strip())
    filtered_tokens = [token.text for token in doc if not token.is_stop and token.text not in [':', ';', ',', '.', '(', ')', '[', ']', '{', '}', '\'', '\"', '-', '~']]

    max_word_diff = 10000000
    for i in range(len(filtered_tokens)):
        for j in range(i+1, len(filtered_tokens)):
            max_word_diff = min(max_word_diff, nlp.vocab[filtered_tokens[i]].similarity(nlp.vocab[filtered_tokens[j]]))
    
    d['max_word_diff'] = max_word_diff

    newData.update({k: d})

newData = json.dumps(newData, indent=4)
with open(out_json, 'w') as f:
    f.write(newData)
