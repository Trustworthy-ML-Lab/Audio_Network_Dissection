import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_lg')

in_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf_llmf.json'
out_json = 'calibration_beats-esc50-unfreeze_esc50_esc50_5_adj_rbf_llmf_allPOS.json'

with open(in_json, 'r') as f:
    data = json.load(f)

newData = {}
all_verbs, all_preps, all_nouns = [], [], []
for k in tqdm(data.keys()):
    d = data[k]
    sentence = ''
    for point in d['highly']:
        sentence = sentence + ' ' + point
    doc = nlp(sentence.strip())
    caption_len = len([token.text for token in doc])

    words = word_tokenize(sentence.lower())
    pos_tags = nltk.pos_tag(words)

    verbs, preps, nouns = [], [], []
    for i in range(len(pos_tags)):
        if pos_tags[i][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: # verb
            if pos_tags[i][0] not in stopwords.words('english'):
                verbs.append(pos_tags[i][0])
        if pos_tags[i][1] in ['IN']: # preposition. Do not use stopwords to filter
            preps.append(pos_tags[i][0])
        if pos_tags[i][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            if pos_tags[i][0] not in stopwords.words('english'):
                nouns.append(pos_tags[i][0])

    verbs =list(set([a for a in verbs]))
    preps = list(set([a for a in preps]))
    nouns = list(set([a for a in nouns]))
    d['verbs'] = verbs
    d['preps'] = preps
    d['nouns'] = nouns
    d['caption_len'] = caption_len
    newData.update({k: d})
    all_verbs.extend(verbs)
    all_preps.extend(preps)
    all_nouns.extend(nouns)
    # print('set(verbs): ', verbs)

all_verbs = set(all_verbs)
all_preps = set(all_preps)
all_nouns = set(all_nouns)
print('all_verbs: ', all_verbs)
# print('all_preps: ', all_preps)
# print('all_nouns: ', all_nouns)
print(len(all_verbs))
print(len(all_preps))
print(len(all_nouns))

newData = json.dumps(newData, indent=4)
with open(out_json, 'w') as f:
    f.write(newData)
