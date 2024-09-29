import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-tn", "--target_name", type=str, default="ast-esc50", choices = ["ast-esc50", "ast-urban8k", "ast-gtzan", "beats-esc50-frozen", "beats-esc50-finetuned",  "beats-urban8k-frozen", "beats-urban8k-finetuned", "beats-gtzan-frozen", "beats-finetuned"], help="model to dissect (target model)")

# For AST
# parser.add_argument("-tl", "--target_layers", type=str, default="layer0_output,layer0_intermediate,layer0_attention_output,layer1_output,layer1_intermediate,layer1_attention_output,layer2_output,layer2_intermediate,layer2_attention_output,layer3_output,layer3_intermediate,layer3_attention_output,layer4_output,layer4_intermediate,layer4_attention_output,layer5_output,layer5_intermediate,layer5_attention_output,layer6_output,layer6_intermediate,layer6_attention_output,layer7_output,layer7_intermediate,layer7_attention_output,layer8_output,layer8_intermediate,layer8_attention_output,layer9_output,layer9_intermediate,layer9_attention_output,layer10_output,layer10_intermediate,layer10_attention_output,layer11_output,layer11_intermediate,layer11_attention_output,fc", help="""Which layer neurons to describe. String list of layer names to describe, separated by comma (no spaces). Follows the naming format of the Pytorch module used""")

# For BEATs
parser.add_argument("-tl", "--target_layers", type=str, default="layer0_1,layer0_2,layer1_1,layer1_2,layer2_1,layer2_2,layer3_1,layer3_2,layer4_1,layer4_2,layer5_1,layer5_2,layer6_1,layer6_2,layer7_1,layer7_2,layer8_1,layer8_2,layer9_1,layer9_2,layer10_1,layer10_2,layer11_1,layer11_2,fc")

parser.add_argument("-pd", "--probing_dataset", type=str, default="esc50", choices = ["esc50", "urban8k", "gtzan"], help="probing dataset to probe the target model")
parser.add_argument("-cs","--concept_set_file", type=str, default="data/concept_set/esc50.txt", help="path to txt file of concept set")
parser.add_argument("-nc", "--network_class_file", type=str, default="data/network_class/esc50.txt", help="path to txt file of network's classification class")

parser.add_argument("--clip_model", default="ViT-B/32", help="CLIP model version to use")
parser.add_argument("--clap_model", default="ViT-B/32", help="CLAP model version to use")
parser.add_argument("--sentence_transformer", default='all-MiniLM-L12-v2', help="sentence transformer to use")

parser.add_argument("--batch_size", type=int, default=1, help="batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which GPU")
parser.add_argument("--seed", default=20, type=int, help="seed number")
parser.add_argument("--num_of_gpus", default = 1, type=int, help="number of available GPUs for vllm")
parser.add_argument("--pool_mode", type=str, default="avg", help="aggregation function for channels")
parser.add_argument("--scoring_func", type=bool, default=False)

parser.add_argument("-dd", "--audio_description_dir", type=str, default="audio_description", help="dir to save audio descriptions")
parser.add_argument("-ad", "--audio_dir", type=str, default="save_audios", help="dir to save audio")
parser.add_argument("-sad", "--save_activation_dir", type=str, default="saved_activations", help="dir to save activation values")
parser.add_argument("-ssd", "--save_summary_dir", type=str, default="summaries", help="dir to save summaries")
parser.add_argument("-sdd", "--save_discriminative_sample_dir", type=str, default="discriminative_samples", help="dir to save discriminative samples")
parser.add_argument("-spd", "--save_prediction_dir", type=str, default="prediction", help="dir to save prediction")
parser.add_argument("-sid", "--save_interpretability_dir", type=str, default='interpretability', help="dir to save interpretability experiments")
parser.add_argument("-dt", "--discriminative_type", type=str, default="highly")
parser.add_argument("-ppt", "--post_process_type", type=str, default="sim")
parser.add_argument("-m", "--mutual_info_threshold", type=float, default=0.6)
parser.add_argument("-k", "--K", type=int, default=5, help="top-K highly/lowly-activated audio")
parser.add_argument("-c", "--clusters", type=int, default=11, help="number of clusters")

# LLM settings
parser.add_argument("--llm", default="meta-llama/Llama-2-13b-chat-hf", help="LLM to use")
parser.add_argument("--top_p", type=float, default=1, help='sampling parameters')
parser.add_argument("--temperature", type=float, default=1, help='sampling parameters')
parser.add_argument("--max_tokens", type=int, default=128, help='sampling parameters')
parser.add_argument("--ICL_topk", type=int, default=1, choices = [1, 5], help="experiments of top5 or top1 accuracy of ICL")

# pruning settings
parser.add_argument("--save_pruning_dir", default="pruning_result", help="dir to save pruning results")
parser.add_argument("--max_pruned_num", type=int, default=3000, help='the maximum of pruned neurons')
parser.add_argument("-pc", "--pruned_concepts", default=["water_drops"], nargs='+', help="what concept to be ablated")
parser.add_argument("-ps", "--pruning_strategy", default="db", choices=["random", "db", "tab", "ocp"], help="what method to decide pruned neurons")
