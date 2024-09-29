import os
import sys

sys.path.append("..")
import json
import random
from collections import defaultdict

from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from args import parser
from data_utils import get_audio_description
from sentence_utils import is_junk_sentence

random.seed(1)

if __name__ == "__main__":
	
	args = parser.parse_args()
	os.chdir("..")

	corpus = []
	filenames = []
	sentence_id_to_cluster_id = dict()
	clustering = defaultdict(lambda: defaultdict(int))

	transformer_model = SentenceTransformer(args.sentence_transformer)
	descriptions = get_audio_description(args.audio_description_dir, args.probing_dataset)

	for filename, description in descriptions.items(): 
		description = sent_tokenize(description)
		description = [s for s in description if not is_junk_sentence(s)]
		ids = [filename for _ in description]
		corpus.extend(description)
		filenames.extend(ids)

	corpus_embeddings = transformer_model.encode(corpus, batch_size=64, show_progress_bar=True)
	
	print("Start clustering")

	clustering_model = KMeans(n_clusters=args.clusters, n_init="auto")
	clustering_model.fit(corpus_embeddings)
	cluster_assignment = clustering_model.labels_
	
	for sentence_id, cluster_id in enumerate(cluster_assignment):
		clustering[filenames[sentence_id]][int(cluster_id)] += 1

	if not os.path.exists(args.save_interpretability_dir):
		os.makedirs(args.save_interpretability_dir)

	with open(os.path.join(args.save_interpretability_dir, f"{args.probing_dataset}_clustering.json"), "w") as f: 
		json.dump(clustering, f, indent=2)
