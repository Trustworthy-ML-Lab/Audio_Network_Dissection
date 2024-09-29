import os

from args import parser
from data_utils import save_discriminative_sample
from sentence_utils import get_basename

if __name__ == "__main__":

	args = parser.parse_args()

	args.target_layers = args.target_layers.split(",")
	args.save_activation_dir = os.path.join(args.save_activation_dir, f'{args.target_name}_{args.probing_dataset}_{get_basename(args.concept_set_file)}')

	dataset = save_discriminative_sample(args.save_discriminative_sample_dir, args.save_activation_dir, args.probing_dataset, args.concept_set_file, args.target_name, args.target_layers, args.K)
