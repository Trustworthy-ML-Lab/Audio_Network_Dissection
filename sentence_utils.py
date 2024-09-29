import ast
import json
import os
import pickle
import re
import string
from collections import Counter, defaultdict

import numpy as np
import spacy

nlp = spacy.load('en_core_web_lg')
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
# from vllm import LLM, SamplingParams
from datasets import Dataset
from nltk import pos_tag, sent_tokenize, word_tokenize
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_basename(path):
    
    filename = os.path.basename(path)
    filename = filename.replace(".txt", "")

    return filename

def cos_similarity(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    # compute cosine similarity
    sim = np.dot(a, b) / (norm(a) * norm(b))
    return sim.item()

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    result = text.translate(translator)
    return result

def clean_word(text):

    replace_char = ["_", "-", ","]
    remove_char = ["(", ")"]

    text = text.lower()
    for char in replace_char:
        text = text.replace(char, " ")
    for char in remove_char: 
        text = text.replace(char, "")

    return text

def has_repeated_substring(s):

    _LENGTH = 8
    seen_substrings = set()
    s = word_tokenize(s)
    for i in range(len(s) - _LENGTH + 1):
        substring = "".join(s[i:i+_LENGTH])
        if substring in seen_substrings:
            return True
        seen_substrings.add(substring)

    return False

def post_process_prediction(response, concepts=None, gt=None, embedding_model=None, K=1, device="cuda"):

    # heuristically find keywords
    keywords = []
    if "Answer:" in response:
        keywords = response.split("Answer:")[-1]
        keywords = keywords.split(",")
    if '"' in response:
        keywords = re.findall(r'"([^"]*)"', response)
    elif "'" in response:
        keywords = re.findall(r"'([^']*)'", response)
    elif "#" in response:
        response = sent_tokenize(response)
        for sub_response in response:
            if "#" in sub_response:
                keywords = sub_response.split("#")[-1]
    if keywords == []: # if failed to heuristically find keywords
        keywords = [response]

    keywords = keywords[:K]
    if len(keywords) > K and keywords[0] != keywords[1]:
        print(f"Warning!! Keywords are more than {K}.", keywords)

    model = SentenceTransformer(embedding_model, device=device)
    model = model.to(device)
    predictions = []

    # determine predictions by cos sim
    for keyword in keywords:
        sentence_template = "This is an audio about {}"
        prediction_sentence = sentence_template.format(keyword)
        prediction_embedding = model.encode(prediction_sentence)
        concept_sentence = [sentence_template.format(concept) for concept in concepts]
        embeddings = model.encode(concept_sentence, batch_size=32)
        
        max = -100
        for concept, embedding in zip(concepts, embeddings):
            similarity = cos_similarity(prediction_embedding, embedding)
            if similarity > max:
                max = similarity
                keyword = concept
        predictions.append(keyword)

    # calculate cos sim
    gt_embeds = model.encode(gt)
    pred_embeds = model.encode(predictions[0])
    cos_sim = cos_similarity(gt_embeds, pred_embeds)

    return predictions, cos_sim

def remove_dummy(adjs):

    dummy = set(['other', 'most', 'some', 'any', 'several', 'many', 'few', 'all', 'each', 'every', 'another', 'both', 'either', 'neither', 'such', 'more', 'less', 'a few', 'a lot', 'several', 'many', 'much', 'little', 'most', 'none', 'no one', 'somebody', 'someone', 'something', 'somewhere', 'used', 'audio', 'best', 'due', 'recorded', 'most', 'various', 'video', 'meant', 'easy', '737-800', 'personal', 'external', 'overall',
    'sound', 'mobile', 'designed', 'well-defined', 'detailed', 'suitable', 'small', 'third', 'second', 'fourth', 'fifth', 'first', 'related', 'different', 'actual', 'kitchen', '*', '2-3', 'everyday', 'common'])
    
    return adjs.difference(dummy)

def is_junk_sentence(sentence):
    junk_words = ["sure", "here are", "this help", "based on", "\n\n1", "let me know", "please", "need any more help"]

    sentence = sentence.lower()
    for word in junk_words:
        if word in sentence:
            return True 
    
    return False

def clean_repeated_substring(descriptions):
    for audio_id, des in descriptions.items():
        if has_repeated_substring(des):
            if "\n" in des: 
                des = des.split("\n")[0]
            else: 
                des = des.split(".")[0]
            descriptions[audio_id] = des
    
    return descriptions

def tokenize_summary(tokenized_summaries, summaries, discriminative_type):
    
    for line in summaries:
        target_layer = line["target_layer"]
        neuron_id = line["neuron_id"]
        neuron_id = str(neuron_id)

        summary = sent_tokenize(line["summary"])

        # Sometimes sent_tokenize won't work
        if len(summary) < 3:
            summary = line["summary"].split("\n")

        summary = [s for s in summary if len(word_tokenize(s)) > 5 and not is_junk_sentence(s)]

        tokenized_summaries[target_layer + "#" + neuron_id][discriminative_type] = summary

def remove_mutual_information(save_summary_dir, probing_dataset, concept_set_file, target_name, embedding_model, device, K=5, mutual_info_threshold=0.5):

    model = SentenceTransformer(embedding_model, device=device)
    # model = model.to(device)

    format_string = "{}"
    summary_file = os.path.join(save_summary_dir, f'{target_name}_{probing_dataset}_{get_basename(concept_set_file)}_{format_string}_top{K}.json')
    highly_summary_file = summary_file.format("highly")
    lowly_summary_file =  summary_file.format("lowly")

    with open(highly_summary_file) as f:
        highly_summaries = json.load(f)
    
    with open(lowly_summary_file) as f: 
        lowly_summaries = json.load(f)

    tokenized_summaries = defaultdict(lambda: defaultdict(dict))
    summaries_embedding = defaultdict(lambda: defaultdict(dict))

    tokenize_summary(tokenized_summaries, highly_summaries, discriminative_type="highly")
    tokenize_summary(tokenized_summaries, lowly_summaries, discriminative_type="lowly")

    for id, object in tokenized_summaries.items():
        for type, summary in object.items():
            embeddings = model.encode(summary, batch_size=32)
            summaries_embedding[id][type] = embeddings

        for summary_a, embedding_a in zip(object["highly"], summaries_embedding[id]["highly"]):
            for summary_b, embedding_b in zip(object["lowly"], summaries_embedding[id]["lowly"]):
                if cos_similarity(embedding_a, embedding_b) > mutual_info_threshold:
                    # Remove this summary if it exists
                    try:
                        tokenized_summaries[id]["highly"].remove(summary_a)
                    except:
                        pass
                    try:
                        tokenized_summaries[id]["lowly"].remove(summary_b)
                    except: 
                        pass

    with open(os.path.join(save_summary_dir, f'removal_{target_name}_{probing_dataset}_{get_basename(concept_set_file)}_top{K}.json'), "w") as f:
        json.dump(tokenized_summaries, f, indent=2)
    
    return tokenized_summaries


def rule_based_adj_filter(processed_summaries, target_name):

    all_adjs = []
    for key, sentences in tqdm(processed_summaries.items()):
        
        summary = " ".join(sentences["highly"])

        words = word_tokenize(summary.lower())
        pos_tags = pos_tag(words)

        adjs = []
        for i in range(len(pos_tags)):
            if pos_tags[i][1] in ['JJ', 'JJR', 'JJS', 'VBN']:
                pre_words = pos_tags[i - 4:i]
                pre_words = [k[0] for k in pre_words]
                if 'no' not in pre_words and 'not' not in pre_words:
                    adjs.append(pos_tags[i][0])

        adjs = set(adjs)
        adjs = list(remove_dummy(adjs))
        all_adjs += adjs

        processed_summaries[key]['adj_after_rbf'] = adjs
    
    with open(f"all_adj_{target_name}.txt", "w") as f:
        print(str(all_adjs), file = f ) 

    return processed_summaries

def llm_based_adj_filter(processed_summary, acoustic_words_file):

    with open(acoustic_words_file, 'r') as f:
        all_words = json.load(f)

    is_acoustic_word = {}
    for word in all_words:
        description = word['response'].lower()
        ans = True if 'yes' in description else False
        is_acoustic_word[word['word']] = ans

    for key, data in tqdm(processed_summary.items()):
        
        adjs = data['adj_after_rbf']
        acoustic_word = []
        for adj in adjs:
            adj = adj.strip('\'')
            if is_acoustic_word[adj] == True:
                acoustic_word.append(adj)

        processed_summary[key]['adj_after_rbf_llmf'] = acoustic_word
    
    return processed_summary

def all_pos_filter(processed_summaries):

    all_verbs, all_preps, all_nouns = [], [], []
    for key, data in tqdm(processed_summaries.items()):
        
        summary = " ".join(data["highly"])

        doc = nlp(summary.strip())
        caption_len = len([token.text for token in doc])

        words = word_tokenize(summary.lower())
        pos_tags = pos_tag(words)

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

        # verbs =list(set([a for a in verbs]))
        # preps = list(set([a for a in preps]))
        # nouns = list(set([a for a in nouns]))
        processed_summaries[key]['verbs'] = verbs
        processed_summaries[key]['preps'] = preps
        processed_summaries[key]['nouns'] = nouns
        processed_summaries[key]['caption_len'] = caption_len
        # all_verbs.extend(verbs)
        # all_preps.extend(preps)
        # all_nouns.extend(nouns)

    # all_verbs = set(all_verbs)
    # all_preps = set(all_preps)
    # all_nouns = set(all_nouns)
    # print('all_verbs: ', all_verbs)
    # print('all_preps: ', all_preps)
    # print('all_nouns: ', all_nouns)
    # print(len(all_verbs))
    # print(len(all_preps))
    # print(len(all_nouns))

    return processed_summaries


