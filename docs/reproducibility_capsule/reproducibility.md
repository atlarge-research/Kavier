# Reproducibility Capsule

## Experiment 1

We run experiment 1 to validate accuracy of Kavier. 

Then, we validate performance of Kavier and compare the simulation
time versus real world time, even at massive scale operation.

#### Experiment 1.1: Accuracy Validation for prefill stage

We want to validate the acccuracy of Kavier for the prefill stage. We will use a trace file that contains the input data for the prefill stage.
- Input: `reproducibility_capsule/experiment1/input/input_prefill_accuracy.csv`
- Setup:
```bash
--llm Llama-3-8B
--gpu A10
--trace reproducibility_capsule/experiment1/input/input_prefill_accuracy.csv 
--kv_cache on
--output_folder reproducibility_capsule/experiment1/output/
```

#### Experiment 1.2: Performance Validation for decode stage

We want to validate the performance of Kavier for the decode stage. We will use a trace file that contains the input data for the decode stage.

- Input: `reproducibility_capsule/experiment1/input/input_decode_accuracy.csv`
- Setup
```bash
--llm Llama-3-8B
--gpu A10
--trace reproducibility_capsule/experiment1/input/input_decode_accuracy.csv 
--kv_cache on
--output_folder reproducibility_capsule/experiment1/output/
```

### Experiment 1.3: Run a very large scale simulation

We want to run a very large scale simulation and see the impact on performance. We want to see how the export rate affects the performance of Kavier.

- Input: `reproducibility_capsule/experiment1/input/marconi_all_traces_only_counts.csv`
- Setup

```bash
--llm Llama-3-8B
--gpu A10
--trace reproducibility_capsule/very_big_data/marconi_all_traces.csv 
--kv_cache on
--output_folder reproducibility_capsule/experiment1/output/
--export_rate [SELECT BETWEEN 0.001, 0.01, 0.1, 1] which means 1ms, 10ms, 100ms, 1s
```

Measurements: 
- 1s export rate takes 10.21s, 10.12s
- 100ms export rate takes 98.16s
- 10ms export rate takes 

---

## Experiment 3 - See different caching policies impacts on performance and performance efficiency
Constants: A10, Llama-3-8B, 1s export rate, prefill-only, LRU evict policy
Variables: prefix size (1024, 2048, 4092, none), cache size (8 prompts, 16 prompts per session)

```bash
--llm Llama-3-8B
--gpu A10
--trace reproducibility_capsule/very_big_data/marconi_full.parquet
--kv_cache on
--output_folder data/output/
--export_rate 1
--flush_size 1000
--prefix_cache_min_tokens [pick 1024, 2048, 4096, 1000000] # 1000000 means no prefix cache
--max_cached_prompts [pick 8, 16]
--cache_scope session
--prefix_cache_policy prefill
```

## Experiment 2 - See differences of KV on vs off
Constants: A10, Llama-3-8B, 1s export rate, prefix caching disabled, opendc 8 models in metamodel 
Variables: kv_cache on vs off

```bash
--llm [Llama-3-8B,  
--gpu A100 
--trace reproducibility_capsule/very_big_data/small.parquet
--kv_cache [pick between on, off]
--output_folder data/output/
--export_rate 1
--flush_size 1000
--prefix_cache_min_tokens 10
--prefix_cache_policy none
```