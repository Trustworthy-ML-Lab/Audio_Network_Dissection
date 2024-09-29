import os

from args import parser
from sentence_utils import get_basename
from utils import save_activations

if __name__ == "__main__":

	args = parser.parse_args()

	args.target_layers = args.target_layers.split(",")
	args.save_activation_dir = os.path.join(args.save_activation_dir, f"{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}")

	save_activations(args)
