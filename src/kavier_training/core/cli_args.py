"""
CLI argument parser for Kavier training simulation.
"""

import argparse


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add training-specific arguments to the parser.
    - model_name (e.g., llama3.1-70b, granite-3.1-8b-instruct)
    - method (full, lora)
    - gpu_model (e.g., NVIDIA-A100-SXM4-80GB)
    - tokens_per_sample (e.g., 512, 1024, 4096, 8192)
    - batch_size (e.g., 8, 16, 32, 64, 128)
    - number_gpus (e.g., 8, 16)
    - number_nodes (e.g., 1, 2)
    
    Args:
        parser: ArgumentParser instance to add arguments to
        
    Returns:
        ArgumentParser with training arguments added
    """
    # ── Required Arguments (from dataset) ─────────────────────────
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name (e.g., llama3.1-70b, granite-3.1-8b-instruct, llama3.1-8b)",
    )
    
    parser.add_argument(
        "--method",
        required=True,
        choices=["full", "lora", "gptq-lora"],
        help="Fine-tuning method: full (all params), lora (low-rank adaptation), or gptq-lora (quantized lora)",
    )
    
    parser.add_argument(
        "--gpu_model",
        required=True,
        help="GPU model (e.g., NVIDIA-A100-SXM4-80GB, NVIDIA-A100-80GB-PCIe)",
    )
    
    parser.add_argument(
        "--tokens_per_sample",
        type=int,
        required=True,
        help="Sequence length in tokens (e.g., 512, 1024, 2048, 4096, 8192)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size per GPU (e.g., 8, 16, 32, 64, 128)",
    )
    
    parser.add_argument(
        "--number_gpus",
        type=int,
        required=True,
        help="Number of GPUs (e.g., 8, 16)",
    )
    
    parser.add_argument(
        "--number_nodes",
        type=int,
        required=True,
        help="Number of nodes (e.g., 1, 2)",
    )

    # ── Metrics of Interest ───────────────────────────────────────
    parser.add_argument(
        "--metrics",
        choices=["performance", "energy", "both"],
        default="both",
        help="Metrics to compute: performance (throughput/time), energy (power), or both",
    )

    # ── Output ────────────────────────────────────────────────────
    parser.add_argument(
        "--output_folder",
        default="src/data/output",
        help="Path to output folder for results",
    )

    return parser


