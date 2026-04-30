"""
Test training simulation against real measured data from ado-sfttrainer.

Filters for:
- Valid runs only
- Single-GPU nodes only (number_gpus == 1)
- Full fine-tuning method only (for now)
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from library.llm import LLM_SPEC_LIBRARY
from library.gpu import GPU_SPEC_LIBRARY
from kavier_training.components.forward_pass import calculate_forward_pass
from kavier_training.components.backward_pass import calculate_backward_pass
from kavier_training.components.optimizer import calculate_optimizer_step


def load_real_data():
    """Load and filter real training data."""
    data_path = Path(__file__).parent.parent.parent / "finetuning-data" / "in" / "curated" / "ado-sfttrainer_valid_runs.csv"
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"\nTotal rows: {len(df)}")
    
    # Filter for single-GPU only
    df_single_gpu = df[df['number_gpus'] == 1].copy()
    print(f"Single-GPU rows: {len(df_single_gpu)}")
    
    # Filter for full fine-tuning only (for now)
    df_full = df_single_gpu[df_single_gpu['method'] == 'full'].copy()
    print(f"Full fine-tuning rows: {len(df_full)}")
    
    return df_full


def simulate_training_step(row):
    """Simulate one training step for a given configuration."""
    model_name = row['model_name']
    gpu_model = row['gpu_model']
    batch_size = int(row['batch_size'])
    seq_length = int(row['tokens_per_sample'])
    
    # Get specs
    if model_name not in LLM_SPEC_LIBRARY:
        return None, f"Model {model_name} not in library"
    if gpu_model not in GPU_SPEC_LIBRARY:
        return None, f"GPU {gpu_model} not in library"
    
    llm = LLM_SPEC_LIBRARY[model_name]
    gpu = GPU_SPEC_LIBRARY[gpu_model]
    
    # Simulate
    forward_time_s, activation_memory_gb = calculate_forward_pass(
        batch_size=batch_size,
        seq_length=seq_length,
        llm=llm,
        gpu=gpu,
    )
    
    backward_time_s, gradient_memory_gb = calculate_backward_pass(
        forward_time_s=forward_time_s,
        llm=llm,
    )
    
    optimizer_time_s, optimizer_memory_gb = calculate_optimizer_step(
        llm=llm,
        gpu=gpu,
    )
    
    # Total (overhead removed - was unused empirical function)
    total_step_time_s = forward_time_s + backward_time_s + optimizer_time_s
    
    # Calculate throughput (tokens per second)
    tokens_per_step = batch_size * seq_length
    predicted_throughput = tokens_per_step / total_step_time_s
    
    return {
        'forward_time_s': forward_time_s,
        'backward_time_s': backward_time_s,
        'optimizer_time_s': optimizer_time_s,
        'total_step_time_s': total_step_time_s,
        'predicted_throughput': predicted_throughput,
    }, None


def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error."""
    if actual == 0:
        return 0.0
    return abs(actual - predicted) / actual * 100


def main():
    print("=" * 80)
    print("Testing Training Simulation Against Real Data")
    print("=" * 80)
    
    # Load data
    df = load_real_data()
    
    if len(df) == 0:
        print("\n❌ No data found matching criteria (single-GPU, full fine-tuning)")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing {len(df)} configurations")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, row in df.iterrows():
        model_name = row['model_name']
        gpu_model = row['gpu_model']
        batch_size = int(row['batch_size'])
        seq_length = int(row['tokens_per_sample'])
        
        print(f"\n{'='*80}")
        print(f"Configuration {idx + 1}/{len(df)}")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"GPU: {gpu_model}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_length}")
        
        # Simulate
        sim_result, error = simulate_training_step(row)
        
        if error:
            print(f"❌ Error: {error}")
            continue
        
        # Get actual measurements
        actual_throughput = row['dataset_tokens_per_second']
        actual_runtime = row['train_runtime']
        
        # Compare
        predicted_throughput = sim_result['predicted_throughput']
        throughput_mape = calculate_mape(actual_throughput, predicted_throughput)
        
        print(f"\n{'Metric':<30} {'Actual':<20} {'Predicted':<20} {'MAPE %':<10}")
        print(f"{'-'*80}")
        print(f"{'Throughput (tokens/s)':<30} {actual_throughput:<20.2f} {predicted_throughput:<20.2f} {throughput_mape:<10.2f}")
        print(f"{'Step time (s)':<30} {sim_result['total_step_time_s']:<20.4f} {'-':<20} {'-':<10}")
        print(f"  {'- Forward':<28} {sim_result['forward_time_s']:<20.4f}")
        print(f"  {'- Backward':<28} {sim_result['backward_time_s']:<20.4f}")
        print(f"  {'- Optimizer':<28} {sim_result['optimizer_time_s']:<20.4f}")
        
        results.append({
            'model_name': model_name,
            'gpu_model': gpu_model,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'actual_throughput': actual_throughput,
            'predicted_throughput': predicted_throughput,
            'throughput_mape': throughput_mape,
        })
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        df_results = pd.DataFrame(results)
        avg_mape = df_results['throughput_mape'].mean()
        median_mape = df_results['throughput_mape'].median()
        min_mape = df_results['throughput_mape'].min()
        max_mape = df_results['throughput_mape'].max()
        
        print(f"\nThroughput Prediction Accuracy:")
        print(f"  Average MAPE: {avg_mape:.2f}%")
        print(f"  Median MAPE:  {median_mape:.2f}%")
        print(f"  Min MAPE:     {min_mape:.2f}%")
        print(f"  Max MAPE:     {max_mape:.2f}%")
        
        print(f"\n{'='*80}")
        
        if avg_mape < 20:
            print("✅ Good accuracy (< 20% MAPE)")
        elif avg_mape < 50:
            print("⚠️  Moderate accuracy (20-50% MAPE)")
        else:
            print("❌ Poor accuracy (> 50% MAPE)")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()


