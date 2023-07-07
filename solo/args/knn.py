import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_knn() -> argparse.Namespace:
    """Parses arguments for offline K-NN.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add knn args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--k", type=int, nargs="+", help="Set to 20 for default value")
    parser.add_argument("--temperature", type=float, nargs="+", help="Set to 0.07 for default value")
    parser.add_argument("--distance_function", type=str, nargs="+", help="`cosine` or `euclidean`")
    parser.add_argument("--feature_type", type=str, nargs="+", help="`backbone` or `projector`")

    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
