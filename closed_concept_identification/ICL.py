import json
import os
import sys

sys.path.append("..")
from vllm import LLM, SamplingParams

from args import parser
from data_utils import get_concept_dataset, get_concept_set
from sentence_utils import post_process_prediction

if __name__ == "__main__":

	args = parser.parse_args()
	os.chdir("..")
	
	if args.ICL_topk == 1:
		prompt_template =  "<s>[INST] <<SYS>>\n\
		You have a set of object classnames: \
		{} \
		\n \
		The following is a description about some audio clips. Based on the description, select a classname out of the above classnames that matches the description most. \
		<</SYS>>\n\n \
		The audio features a car meowing: All of the clips contain the sound of a cat meowing. Loud sound: These clips are all of loud sound but with varying degrees of intensity. Repetitive barking: Clips 1 and 4 are repetitive, with the cat meowing multiple times in each clip. Poor audio quality: All clips have poor audio quality, with either distortion, muffling, or apparent background noises. \
		[/INST] \
		We know these clips are about the class 'cat' in the concept set. We can get this answer since the description mentions All of the clips contain the sound of a cat meowing'. \n \
		Answer: cat \
		</s><s>[INST] \
		They all feature a person snoring loudly. The snoring starts off slow and gets louder over time. The audio is recorded in mono. There are no other sounds in the background. The snoring is described as loud and intense. The audio clips differ in the following ways. The first clip features a man snoring, while the second and fourth clips feature a person snoring (gender not specified). The third clip features a zombie growling and snarling, while the other clips only feature snoring. The third clip is described as scary and creepy, while the other clips are not. The third clip is intended for use in a horror movie or zombie video game, while the other clips do not have specific intended uses stated. The third clip is of poor quality, while the other clips are not specified as such. \
		[/INST] \
		Based on the description, the most suitable classname for the audio clips would be 'snoring' or 'zombie_growling_and_snarling'. Both of these classnames match the description of loud sounds with a strong emotional impact, specifically fear and terror. But 'zombie_growling_and_snarling' is not in the given classname set. So the answer is 'snoring' \
		Answer: soring \
		</s><s>[INST] \
		{} \
		[/INST]"
	else:
		prompt_template =  "<s>[INST] <<SYS>>\n\
		You have a set of object classnames: \
		{} \
		\n \
		The following is a description about some audio clips. Based on the description, select 5 classnames out of the above classnames that matches the description most and given the reason. There should be exactly 5 answers.  \
		<</SYS>>\n\n \
		The audio features a cat meowing: All of the clips contain the sound of a cat meowing. Loud sound: These clips are all of loud sound but with varying degrees of intensity. Repetitive barking: Clips 1 and 4 are repetitive, with the cat meowing multiple times in each clip. Poor audio quality: All clips have poor audio quality, with either distortion, muffling, or apparent background noises. \
		[/INST] \
		We know these clips are about the class 'cat' in the concept set. We can get this answer since the description mentions All of the clips contain the sound of a cat meowing'. \n \
		Answer: cat, sheep, snoring, cow, coughing \
		</s><s>[INST] \
		They all feature a person snoring loudly. The snoring starts off slow and gets louder over time. The audio is recorded in mono. There are no other sounds in the background. The snoring is described as loud and intense. The audio clips differ in the following ways. The first clip features a man snoring, while the second and fourth clips feature a person snoring (gender not specified). The third clip features a zombie growling and snarling, while the other clips only feature snoring. The third clip is described as scary and creepy, while the other clips are not. The third clip is intended for use in a horror movie or zombie video game, while the other clips do not have specific intended uses stated. The third clip is of poor quality, while the other clips are not specified as such. \
		[/INST] \
		Based on the description, the most suitable classname for the audio clips would be 'snoring' or 'zombie_growling_and_snarling'. Both of these classnames match the description of loud sounds with a strong emotional impact, specifically fear and terror. But 'zombie_growling_and_snarling' is not in the given classname set. So the answer is 'snoring' \
		Answer: snoring, zombie_growling_and_snarling, breathing, footsteps, coughing\
		</s><s>[INST] \
		{} \
		[/INST]"

	args.target_layers = args.target_layers.split(",")
	dataset = get_concept_dataset(args.save_summary_dir, args.probing_dataset, args.concept_set_file, args.target_name, args.target_layers, args.network_class_file, prompt_template, K=args.K)

	llm = LLM(model=args.llm, gpu_memory_utilization=0.9)
	sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, max_tokens=512)

	results = []
	inputs = [line["text"] for line in dataset]
	outputs = llm.generate(inputs, sampling_params)

	for inp, out in zip(dataset, outputs):
		target_layer = inp["target_layer"]
		neuron_id = inp["neuron_id"]
		neuron_label = inp["neuron_label"]
		print(f'-------- {target_layer} # {neuron_id}  --------')

		response = out.outputs[0].text
		results.append({"response": response,
						"target_layer": target_layer,
						"neuron_id": neuron_id,
						"neuron_label": neuron_label})

		if target_layer == "fc":
			print("gt label: ", neuron_label)
			print(response)

		print("=" * 20)
	
	# calculate accuracies
	concepts = get_concept_set(args.concept_set_file)
	
	if not os.path.exists(args.save_prediction_dir):
		os.makedirs(args.save_prediction_dir)

	is_correct, prediction, cos_sim = [], [], []
	for line in results:
		prediction, sim = post_process_prediction(line["response"], concepts=concepts, gt=line["neuron_label"], embedding_model=args.sentence_transformer, K=args.ICL_topk)
		
		is_correct.append(1) if line["neuron_label"] in prediction else is_correct.append(0)
		cos_sim.append(sim)

	print("accuracy: ", round(sum(is_correct) / len(is_correct) * 100, 2))
	print("cos sim : ", round(sum(cos_sim) / len(cos_sim), 2))
	
	# save prediction
	prediction_file = os.path.join(args.save_prediction_dir, f'{args.target_name}_{args.probing_dataset}_{args.concept_set_file.split("/")[-1].split(".txt")[0]}_{args.K}_{args.ICL_topk}.json')
	
	with open(prediction_file, "w") as f:
		json.dump(results, f, indent=2)