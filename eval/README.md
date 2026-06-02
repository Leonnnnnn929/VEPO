# VEPO Evaluation

## Overview

This directory contains the evaluation pipeline for VEPO-trained models. The evaluation covers 6 multimodal math/vision benchmarks:

| Benchmark | Type | Metric | Evaluation Method |
|-----------|------|--------|-------------------|
| Geometry3K (Geo3K) | Geometry reasoning | Accuracy | Rule-based (boxed answer extraction) |
| MathVista | Visual math | Accuracy | LLM-as-Judge |
| MathVerse | Visual math | Accuracy | LLM-as-Judge |
| MathVision | Visual math | Accuracy | LLM-as-Judge |
| WeMath | Visual math | Accuracy | LLM-as-Judge |
| HalluBench | Hallucination | Accuracy | Rule-based (yes/no extraction) |


## Model merge
python scripts/model_merger.py --local_dir YOUR_CHECKPOINT_DIR/actor

## Run evaluation
bash run_eval.sh


## Data Preparation

### Directory Structure

The evaluation data should be organized as follows:

```
eval/data/
├── geometry3k/
│   └── test/
│       ├── 2401/
│       │   ├── img_diagram.png
│       │   └── data.json
│       ├── 2402/
│       │   └── ...
│       └── ...
├── mathvista/
│   └── images/
│       └── ...
├── mathverse/
│   ├── testmini.json
│   └── images/
│       └── ...
├── mathvision/
│   ├── MathVision.tsv
│   └── images/
│       └── ...
├── wemath/
│   ├── testmini.json
│   └── images/
│       └── ...
└── hallubench/
    ├── HallusionBench.json
    └── images/
        └── ...
```

### Data Sources

- **Geometry3K**: Download from [Geometry3K dataset](https://github.com/lupantech/InterGPS)
- **MathVista**: Auto-downloaded via HuggingFace datasets (`AI4Math/MathVista`). Images need to be placed in `eval/data/mathvista/images/`.
- **MathVerse**: Download from [MathVerse](https://github.com/ZrrSkyworker/MathVerse)
- **MathVision**: Download from [MathVision](https://github.com/mathvision-cuhk/MathVision)
- **WeMath**: Download from [WeMath](https://github.com/We-Math/We-Math)
- **HalluBench**: Download from [HallusionBench](https://github.com/tianyi-lab/HallusionBench)

## LLM-as-Judge Configuration

For MathVista, MathVerse, MathVision, and WeMath, we use an LLM API to judge answer correctness. Configure one of the following:

### Option 1: OpenAI-compatible API (recommended)

```bash
export LLM_EVAL_BASE_URL="https://api.openai.com/v1"
export LLM_EVAL_API_KEY="sk-..."
export LLM_EVAL_MODEL="gpt-4o-mini"
```

### Option 2: Google Gemini API

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

### Rate Limiting

To avoid API rate limits:

```bash
export OPENROUTER_MAX_CONCURRENT_REQUESTS=2   
export OPENROUTER_MIN_INTERVAL=0.8            
export OPENROUTER_MAX_RETRIES=8               
```

## Usage

### Full Evaluation

```bash
bash run_eval.sh
```
