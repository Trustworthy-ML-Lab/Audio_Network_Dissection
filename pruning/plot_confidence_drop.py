import json
import os
import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import pandas as pd

from args import parser
from data_utils import get_cls_label, mean

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("../")

	for pruned_concept in args.pruned_concepts:

		result_file = os.path.join(args.save_pruning_dir, f"concept-{args.target_name}-{args.pruning_strategy}.json")

		acc_before_pruning = []
		acc_after_pruning = []

		classes = get_cls_label(args.network_class_file)

		masked_count_list = []
		with open(result_file) as f: 
			all = json.load(f)
			obj = all[pruned_concept]

			confidence_before_avg = []
			confidence_avg = []

			for cls_idx, cls_name in enumerate(classes):
				confidence_before = all["null"]["confidence"][cls_name]
				confidence_after = obj["confidence"][cls_name]
				
				# different samples
				c_b_avg = []
				c_a_avg = []
				for c_b, c_a in zip(confidence_before, confidence_after):
					c_b_avg.append(c_b[cls_idx])
					c_a_avg.append(c_a[cls_idx])

				confidence_before_avg.append(mean(c_b_avg))
				confidence_avg.append(mean(c_a_avg))

			prune_count = obj['masked_count']

		classes = [cls.replace("_", " ") for cls in classes]
		acc_before_pruning = confidence_before_avg
		acc_after_pruning = confidence_avg

		difference = [b - a for a, b in zip(acc_before_pruning, acc_after_pruning)]
		zipped = zip(difference, acc_before_pruning, acc_after_pruning, classes)
		zipped = sorted(zipped, reverse=False)
		difference, acc_before_pruning, acc_after_pruning, classes = zip(*zipped)

		df = pd.DataFrame({
			'Classes': classes,
			'Before pruning': acc_before_pruning,
			'After pruning':acc_after_pruning,
		})

		plt.xticks(rotation=90)

		# Sample data
		x = classes # Common x-values
		y1 = acc_after_pruning  # First set of y-values
		y2 = acc_before_pruning  # Second set of y-values

		# Plotting the points
		plt.figure(figsize=(30, 6))
		plt.scatter(x, y2, color='black',marker='o', s=80, label='confidence before pruning')  # Points for the first set
		plt.scatter(x, y1, color='black', marker='o', facecolors='none', s=90, label='confidence after pruning')  # Points for the second set

		# Connecting points with the same x-value
		for i in range(len(x)):
			if y1[i] > y2[i]:
				color = "#0047AB"
			else: 
				color = "#FF5733"
			if abs(y2[i] - y1[i]) > 0.1:
				plt.annotate('', xy=(x[i], y1[i]), xytext=(x[i], y2[i]),
							arrowprops=dict(facecolor='green', color=color, arrowstyle="->", lw=3, mutation_scale=40))

		# Adding titles and labels
		plt.xlabel('classes', fontsize=36)
		plt.ylabel('confidence', fontsize=36)
		plt.xticks(rotation=30, fontsize=20,  ha='right')
		plt.yticks(fontsize=28)
		plt.legend(loc="lower right",  fontsize="30", ncol=2)

		plt.gca().get_xticklabels()[0].set_color('red') 
		plt.gca().get_xticklabels()[1].set_color('red') 
		plt.gca().get_xticklabels()[2].set_color('red') 
		plt.gca().get_xticklabels()[3].set_color('red') 
		plt.gca().get_xticklabels()[4].set_color('red') 
		plt.gca().get_xticklabels()[4].set_fontsize(22) 
		plt.gca().get_xticklabels()[4].set_weight("bold")

		# Adjust layout to prevent overlap of titles
		plt.tight_layout()
		plt.savefig(os.path.join(args.save_pruning_dir, f'{args.pruning_strategy}_{args.probing_dataset}_{pruned_concept}.pdf'), format="pdf")
		plt.show()
