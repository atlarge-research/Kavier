"""
Test training simulation components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from library.llm_library import LLM_SPEC_LIBRARY
from library.gpu_library import GPU_SPEC_LIBRARY
from simulator.training.forward_pass import calculate_forward_pass
from simulator.training.backward_pass import calculate_backward_pass
from simulator.training.optimizer import calculate_optimizer_step


def test_forward_pass():
    """Test forward pass calculation."""
    print("\n=== Testing Forward Pass ===")
    
    # Use Llama-3-8B and A100-80GB
    llm = LLM_SPEC_LIBRARY["Llama-3-8B"]
    gpu = GPU_SPEC_LIBRARY["A100-80GB"]
    
    batch_size = 4
    seq_length = 512
    
    forward_time_s, activation_memory_gb = calculate_forward_pass(
        batch_size=batch_size,
        seq_length=seq_length,
        llm=llm,
        gpu=gpu,
    )
    
    print(f"Model: {llm.name}")
    print(f"GPU: {gpu.name}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Total tokens: {batch_size * seq_length}")
    print(f"\nForward pass time: {forward_time_s:.4f} seconds ({forward_time_s * 1000:.2f} ms)")
    print(f"Activation memory: {activation_memory_gb:.2f} GB")
    
    # Sanity checks
    assert forward_time_s > 0, "Forward time should be positive"
    assert activation_memory_gb > 0, "Activation memory should be positive"
    assert forward_time_s < 10, "Forward time seems too large"
    assert activation_memory_gb < 100, "Activation memory seems too large"
    
    print("✅ Forward pass test passed")
    return forward_time_s, activation_memory_gb


def test_backward_pass(forward_time_s):
    """Test backward pass calculation."""
    print("\n=== Testing Backward Pass ===")
    
    llm = LLM_SPEC_LIBRARY["Llama-3-8B"]
    
    backward_time_s, gradient_memory_gb = calculate_backward_pass(
        forward_time_s=forward_time_s,
        llm=llm,
    )
    
    print(f"Model: {llm.name}")
    print(f"Model parameters: {llm.m_params / 1e9:.1f}B")
    print(f"\nBackward pass time: {backward_time_s:.4f} seconds ({backward_time_s * 1000:.2f} ms)")
    print(f"Gradient memory: {gradient_memory_gb:.2f} GB")
    print(f"Backward/Forward ratio: {backward_time_s / forward_time_s:.2f}x")
    
    # Sanity checks
    assert backward_time_s > 0, "Backward time should be positive"
    assert gradient_memory_gb > 0, "Gradient memory should be positive"
    assert abs(backward_time_s / forward_time_s - 2.0) < 0.01, "Backward should be ~2x forward"
    
    print("✅ Backward pass test passed")
    return backward_time_s, gradient_memory_gb


def test_optimizer_step():
    """Test optimizer step calculation."""
    print("\n=== Testing Optimizer Step ===")
    
    llm = LLM_SPEC_LIBRARY["Llama-3-8B"]
    gpu = GPU_SPEC_LIBRARY["A100-80GB"]
    
    optimizer_time_s, optimizer_memory_gb = calculate_optimizer_step(
        llm=llm,
        gpu=gpu,
    )
    
    print(f"Model: {llm.name}")
    print(f"GPU: {gpu.name}")
    print(f"GPU memory bandwidth: {gpu.bandwidth_bps / 1e9:.0f} GB/s")
    print(f"\nOptimizer step time: {optimizer_time_s:.4f} seconds ({optimizer_time_s * 1000:.2f} ms)")
    print(f"Optimizer memory (states): {optimizer_memory_gb:.2f} GB")
    
    # Sanity checks
    assert optimizer_time_s > 0, "Optimizer time should be positive"
    assert optimizer_memory_gb > 0, "Optimizer memory should be positive"
    assert optimizer_time_s < 1, "Optimizer time seems too large"
    
    print("✅ Optimizer step test passed")
    return optimizer_time_s, optimizer_memory_gb


def test_full_training_step():
    """Test complete training step."""
    print("\n=== Testing Full Training Step ===")
    
    llm = LLM_SPEC_LIBRARY["Llama-3-8B"]
    gpu = GPU_SPEC_LIBRARY["A100-80GB"]
    
    batch_size = 4
    seq_length = 512
    
    # Forward pass
    forward_time_s, activation_memory_gb = calculate_forward_pass(
        batch_size=batch_size,
        seq_length=seq_length,
        llm=llm,
        gpu=gpu,
    )
    
    # Backward pass
    backward_time_s, gradient_memory_gb = calculate_backward_pass(
        forward_time_s=forward_time_s,
        llm=llm,
    )
    
    # Optimizer step
    optimizer_time_s, optimizer_memory_gb = calculate_optimizer_step(
        llm=llm,
        gpu=gpu,
    )
    
    # Total
    total_time_s = forward_time_s + backward_time_s + optimizer_time_s
    total_memory_gb = activation_memory_gb + gradient_memory_gb + optimizer_memory_gb
    
    # Model memory (parameters in fp16)
    model_memory_gb = (llm.m_params * llm.p_bytes) / (1024**3)
    total_memory_with_model_gb = total_memory_gb + model_memory_gb
    
    print(f"\n{'='*60}")
    print(f"FULL TRAINING STEP SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {llm.name} ({llm.m_params / 1e9:.1f}B parameters)")
    print(f"GPU: {gpu.name} ({gpu.memory_gb:.0f} GB)")
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    print(f"\n{'Component':<20} {'Time (ms)':<15} {'Memory (GB)':<15}")
    print(f"{'-'*50}")
    print(f"{'Forward pass':<20} {forward_time_s * 1000:>10.2f} ms   {activation_memory_gb:>10.2f} GB")
    print(f"{'Backward pass':<20} {backward_time_s * 1000:>10.2f} ms   {gradient_memory_gb:>10.2f} GB")
    print(f"{'Optimizer step':<20} {optimizer_time_s * 1000:>10.2f} ms   {optimizer_memory_gb:>10.2f} GB")
    print(f"{'-'*50}")
    print(f"{'TOTAL':<20} {total_time_s * 1000:>10.2f} ms   {total_memory_gb:>10.2f} GB")
    print(f"\n{'Model parameters':<20} {'':>15} {model_memory_gb:>10.2f} GB")
    print(f"{'TOTAL + MODEL':<20} {'':>15} {total_memory_with_model_gb:>10.2f} GB")
    print(f"\n{'GPU Memory Available':<20} {'':>15} {gpu.memory_gb:>10.2f} GB")
    
    if total_memory_with_model_gb > gpu.memory_gb:
        print(f"\n⚠️  WARNING: Total memory ({total_memory_with_model_gb:.2f} GB) exceeds GPU memory ({gpu.memory_gb:.0f} GB)")
    else:
        print(f"\n✅ Memory fits in GPU ({total_memory_with_model_gb:.2f} GB / {gpu.memory_gb:.0f} GB)")
    
    # Throughput calculation
    tokens_per_step = batch_size * seq_length
    throughput_tokens_per_sec = tokens_per_step / total_time_s
    
    print(f"\nThroughput: {throughput_tokens_per_sec:.2f} tokens/second")
    print(f"{'='*60}")
    
    print("\n✅ Full training step test passed")


if __name__ == "__main__":
    print("Testing Training Simulation Components")
    print("=" * 60)
    
    # Test individual components
    forward_time_s, _ = test_forward_pass()
    test_backward_pass(forward_time_s)
    test_optimizer_step()
    
    # Test full training step
    test_full_training_step()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

# Made with Bob
