import os
import sys
from collections import defaultdict

sys.path.append("../")
from args import parser
from data_utils import get_cls_label, mean, read_json

if __name__ == "__main__":
	
	args = parser.parse_args()
	os.chdir("../")

	result_file = os.path.join(args.save_pruning_dir, f"class-{args.target_name}-{args.pruning_strategy}.json")
	
	results = read_json(result_file)

	classes = get_cls_label(args.network_class_file)

	# confidence on ablating class samples 
	before_confidence = {}
	after_confidence = {}
	neuron_number = {}

	# confidence on remaining class samples
	remaining_class_before_confidence = defaultdict(list)
	remaining_class_after_confidence = defaultdict(list)

	origin = results["null"]
	origin_acc = origin["correct"] / origin["total"]
	origin_confidence_by_class = origin["confidence"]
	
	for cls_name, object in results.items():

		if cls_name == "null":
			continue

		before_confidence[cls_name] = mean([logit[classes.index(cls_name)] for logit in origin_confidence_by_class[cls_name]])
		after_confidence[cls_name] = mean([logit[classes.index(cls_name)] for logit in object["confidence"][cls_name]])

		for cursor, remaining_cls_name in enumerate(classes):
			if remaining_cls_name == cls_name:
				continue
			remaining_class_before_confidence[cls_name].append(mean([logit[classes.index(remaining_cls_name)] for logit in origin_confidence_by_class[remaining_cls_name]]))
			remaining_class_after_confidence[cls_name].append(mean([logit[classes.index(remaining_cls_name)] for logit in object["confidence"][remaining_cls_name]]))
	
		remaining_class_before_confidence[cls_name] = mean(remaining_class_before_confidence[cls_name])
		remaining_class_after_confidence[cls_name] = mean(remaining_class_after_confidence[cls_name])
		neuron_number[cls_name] = object["masked_count"]

	ablating_class_before = [value for value in before_confidence.values()]
	ablating_class_after = [value for value in after_confidence.values()]
	ablating_delta = (sum(ablating_class_after) - sum(ablating_class_before)) / len(ablating_class_before)

	remaining_class_before = [value for value in remaining_class_before_confidence.values()]
	remaining_class_after = [value for value in remaining_class_after_confidence.values()]
	remaining_delta = (sum(remaining_class_after) - sum(remaining_class_before)) / len(remaining_class_before)

	neuron_number = [value for value in neuron_number.values()]

	print("ablating class before", mean(ablating_class_before))
	print("ablating class after", mean(ablating_class_after))  
	print("ablating class delta", ablating_delta) 
	print("neuron_number", mean(neuron_number))
	print("remaining class before", mean(remaining_class_before))
	print("remaining class after", mean(remaining_class_after))  
	print("remaining class delta", remaining_delta) 
