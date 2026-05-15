import argparse


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--method", required=True, choices=["full", "lora", "gptq-lora"])
    parser.add_argument("--gpu_model", required=True)
    parser.add_argument("--tokens_per_sample", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--number_gpus", type=int, required=True)
    parser.add_argument("--number_nodes", type=int, required=True)
    return parser
