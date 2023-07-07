import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_ir() -> argparse.Namespace:
    """Parses arguments for offline Image Retrieval

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add ir args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--distance_function", type=str, nargs="+", help="`cosine` or `euclidean`")
    parser.add_argument("--ratios", type=float, nargs="+", help="ratios to calculate between 0 and 1 ")
    parser.add_argument("--feature_type", type=str, nargs="+", help="`backbone` or `projector`")

    # dataset
    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
        "custom",
        "rhid-val",
        "rhid-test",
        "hotels50k-test",
    ]

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, type=str, required=True)
    parser.add_argument("--data_path", type=str, help="dataset path")
    parser.add_argument("--vals", type=str, nargs="+", help="Different val dir names")
    parser.add_argument("--train_name", type=str, default='trainval', help="Not going to be used, but should be given")
    parser.add_argument(
        "--data_format", default="image_folder", choices=["image_folder", "dali", "h5"]
    )

    # percentage of data used from training, leave 1.0 to use all data available
    parser.add_argument("--data_fraction", default=-1.0, type=float)
    parser.add_argument("--test", action='store_true')


    # add shared arguments
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
