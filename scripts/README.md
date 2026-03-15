# Hardware-Aware NAS for Edge Devices — Scripts Overview

This directory contains the core Python scripts for the Hardware-Aware Neural Architecture Search (NAS) project on the Tiny-ImageNet-200 dataset. The workflow is divided into 9 logical phases, spanning from data exploration to model search, fine-tuning, and final edge-device optimization.

## System Architecture & Workflow Diagram

```mermaid
graph TD
    classDef data fill:#27ae60,stroke:#fff,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef script fill:#2980b9,stroke:#fff,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef model fill:#8e44ad,stroke:#fff,stroke-width:2px,color:#fff,rx:5,ry:5;

    subgraph phase1 ["Phase 1: Data Exploration"]
        EDA["eda.py"]:::script
        Stats["dataset_stats.json"]:::data
    end

    subgraph phase2 ["Phase 2: Preprocessing"]
        DP["data-processing.py"]:::script
        Aug["Augmentation\nLoaders"]:::data
    end

    subgraph phase3 ["Phase 3: Baselines"]
        MT["model-training.py"]:::script
        BModels[("MobileNetV2\nShuffleNetV2\nEfficientNet-B0")]:::model
    end

    subgraph phase46 ["Phases 4-6: NAS & Search"]
        HW["hardware-aware.py"]:::script
        LUT["latency_lut.json"]:::data
        SNet[("supernet.pth")]:::model
        BArch["best_arch.json"]:::data
    end

    subgraph phase_nas ["NAS Fine-Tuning"]
        NAS["nas.py"]:::script
        NASModel[("nas_best.pth")]:::model
    end

    subgraph phase79 ["Phases 7-9: Eval"]
        EVAL["evaluation.py"]:::script
        OptModels[("Quantized\nPruned\nONNX")]:::model
        Charts["Eval Charts"]:::data
    end

    EDA -.->|"RGB mean/std"| Stats
    Stats -.-> DP
    DP -.-> Aug
    Aug -.-> MT
    MT -.-> BModels
    Aug -.-> HW
    HW -->|"1. Extract LUT"| LUT
    HW -->|"2. One-Shot"| SNet
    HW -->|"3. Evolution"| BArch
    Aug -.-> NAS
    BArch -.-> NAS
    NAS -.-> NASModel
    BModels -.-> EVAL
    NASModel -.-> EVAL
    EVAL --> OptModels
    EVAL --> Charts
```

## Scripts Description

| Script                   | Purpose                                                                                                                                                                                                                                 | Key Outputs                                                              |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **`eda.py`**             | **Phase 1:** Dataset integrity checks, class distribution, and pixel intensity statistics. Computes the normalization factors (mean/std).                                                                                               | `dataset_stats.json`, EDA plots                                          |
| **`data-processing.py`** | **Phase 2:** Defines the standard data augmentation strategy (Crop, Flip, Jitter, Erase) and creates efficient PyTorch DataLoaders.                                                                                                     | `augmentation_preview.png`, `dataloader_benchmark.json`                  |
| **`model-training.py`**  | **Phase 3:** Trains standard lightweight baselines (MobileNetV2, ShuffleNetV2, EfficientNet-B0) to serve as reference points.                                                                                                           | `{baseline}_best.pth`, baseline metrics & plots                          |
| **`hardware-aware.py`**  | **Phases 4–6:** The core NAS algorithm. Defines the cell-based search space, builds a latency lookup table (LUT), trains a One-Shot Supernet, and runs an Evolutionary Search to find the optimal architecture within a latency budget. | `latency_lut.json`, `supernet_final.pth`, `best_arch.json`, Pareto front |
| **`nas.py`**             | **NAS Extraction & Fine-tuning:** Takes the optimal architecture (`best_arch.json`), instantiates it as a leaner standalone model, and trains it from scratch to convergence.                                                           | `nas_best_finetuned.pth`                                                 |
| **`evaluation.py`**      | **Phases 7–9:** Final optimization and benchmark. Applies Post-Training Quantization (INT8), Magnitude Pruning, exports to ONNX, and benchmarks all models (baselines + NAS) generating final comparative charts.                       | Quantized/ONNX models, `final_comparison.json`, scatter/bubble charts    |

## Usage Pipeline

To reproduce the entire pipeline, execute the scripts sequentially:

1. `python eda.py`
2. `python data-processing.py`
3. `python model-training.py`
4. `python hardware-aware.py`
5. `python nas.py`
6. `python evaluation.py`
