"""Argparse flags for the training model, shared by ``kavier training`` and ``kavier carbon --from-training``."""

import argparse

from kavier.sdk.training.core.engine import known_methods


def add_training_job_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the model/method/GPU/sizing flags shared by the training and carbon (from-training) commands."""
    parser.add_argument("--model_name")
    parser.add_argument("--method", choices=known_methods())
    parser.add_argument("--gpu_model")
    parser.add_argument("--tokens_per_sample", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--number_gpus", type=int)
    parser.add_argument("--number_nodes", type=int)
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=None,
        help="Total tokens to train over (sets runtime); or use --epochs + --dataset_tokens.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Passes over the dataset; with --dataset_tokens derives total_tokens.",
    )
    parser.add_argument(
        "--dataset_tokens", type=int, default=None, help="Tokens in one epoch of the dataset (used with --epochs)."
    )
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all ``kavier training`` arguments to ``parser`` and return it."""
    parser.add_argument(
        "--config",
        default=None,
        help="YAML file of {arg_name: value} applied as defaults (explicit flags still override).",
    )
    # The job flags are required without --input_csv; the subcommand checks this.
    parser.add_argument("--input_csv", default=None, help="Simulate every row of this CSV instead of a single config")
    add_training_job_args(parser)
    return parser
