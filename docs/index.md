# Kavier

Kavier is a physics-driven simulator able to simulate LLM ecosystems under inference workloads[^bsc],
extended to support physics-driven simulation of LLM fine-tuning workloads[^msc].

It predicts the performance, sustainability, and efficiency of LLM ecosystems:

* Performance: inference latencies, training throughput, GPU utilization and Model FLOPs Utilization (MFU)
* Sustainability: energy consumption, carbon emissions
* Efficiency: financial and energy cost per token or sample

Kavier ports one of OpenDC's energy models and reproduces OpenDC's energy predictions, four orders of
magnitude faster than OpenDC[^msc].

## Install

```bash
pip install kavier
kavier --help
```

## Pages

* [Fine-tuning model](training.md): the analytical model and its calibration.
* [Usage](usage.md): CLI and Python API.
* [Cluster simulator](cluster-usage.md): FIFO/backfill queuing over a fixed cluster.

## Cite

```bibtex
@software{kavier,
  author = {Nicolae, Radu and Lotito, Daniele and Trivedi, Animesh and Donkervliet, Jesse and Iosup, Alexandru},
  title  = {Kavier: Simulating the Performance, Sustainability, and Efficiency of LLM Ecosystems under Inference and Training},
  year   = {2026},
  doi    = {10.5281/zenodo.23110996},
  url    = {https://github.com/atlarge-research/Kavier}
}
```

The metadata is in [`CITATION.cff`](https://github.com/atlarge-research/Kavier/blob/master/CITATION.cff).

[^bsc]: R. Nicolae, A. Iosup, A. Trivedi, J. Donkervliet. *Kavier: Exploring Performance, Sustainability, and
    Efficiency of LLM Ecosystems under Inference through Cache-Aware Discrete-Event Simulation.* BSc thesis,
    Vrije Universiteit Amsterdam, 2025. [PDF](https://atlarge-research.com/pdfs/2025-15-07_bsc_thesis_radu_nicolae.pdf)
[^msc]: R. Nicolae, D. Lotito, A. Iosup. *Coastline: Exploring the impact of multi-objective, context-aware recommenders
    on performance and sustainability of datacenters under LLM fine-tuning workloads.* MSc thesis, Vrije Universiteit
    Amsterdam, 2026.
