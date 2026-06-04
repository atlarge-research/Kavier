import argparse


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Required for single-config runs; alternatively pass --input_csv to simulate
    # every row of a CSV (e.g. data/input/input_example.csv). Enforced in cli.main.
    parser.add_argument("--input_csv", default=None, help="Simulate every row of this CSV instead of a single config")
    parser.add_argument("--model_name")
    parser.add_argument("--method", choices=["full", "lora", "gptq-lora"])
    parser.add_argument("--gpu_model")
    parser.add_argument("--tokens_per_sample", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--number_gpus", type=int)
    parser.add_argument("--number_nodes", type=int)
    parser.add_argument("--total_tokens", type=int, default=None)
    parser.add_argument("--opendc_output_dir", default=None)
    parser.add_argument("--opendc_task_id", type=int, default=0)
    parser.add_argument("--opendc_submission_time_ms", type=int, default=0)
    return parser
