# ID2223_finetuning

This project fine-tunes Llama 3.2B 4-bit instruct models on two datasets (FineTome-100k and Finance-500k) using LoRA adapters to evaluate whether specialized fine-tuning delivers meaningful performance improvements. We tested the models against 200 diverse test cases and compared them with 1B and 3B base models. Results show marginal improvements from fine-tuning, with model size being the dominant factor in performance. The following claim quoted from the [reference](https://huggingface.co/blog/mlabonne/sft-llama3) unsloth finetuning blog
> The resulting FineTome is an ultra-high quality dataset that includes conversations, reasoning problems, function calling, and more.
were not strongly supported by our evaluation, as we had assumed it would be much better at those tasks.

## User Interface
UI of the chatbot for communication with the fine-tuned LLM's can be found [here](https://huggingface.co/spaces/fattha-kth/id2223-finetuning)

## Datasets

- **[FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)**: Curated dataset with conversations, reasoning problems, and function calling examples
- **[Finance-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)**: Specialized financial domain dataset

## Models Evaluated

### Fine-Tuned Models
- [Unsloth Llama 3.2B 4-bit instruct + LoRA adapters (FineTome)](https://huggingface.co/hellstone1918/Llama-3.2-3B-basic-lora-model)
- [Unsloth Llama 3.2B 4-bit instruct + RS-LoRA adapters (FineTome)](https://huggingface.co/hellstone1918/Llama-3.2-3B-rslora-model)
- [Unsloth Llama 3.2B 4-bit instruct + LoRA adapters (Finance)](https://huggingface.co/hellstone1918/Llama-3.2-3B-finance-lora-model-v6)

### Baseline Model
- Llama 3B base model

## Evaluation Framework

We designed a comprehensive evaluation across multiple test methodologies:

| Category | Description | Number of Cases |
|----------|-------------|-----------------|
| **Mathematical/Logic Problems for Reasoning** | Arithmetic computations and word problems | ~25 |
| **Function Calling** | Understanding function signatures and proper formatting | ~40 |
| **Situational Inference** | Context-based reasoning and implicit information extraction | ~25 |
| **Multiple Choice Questions** | General and domain-specific knowledge assessment | ~25 |
| **Prometheus v2.0 Evaluation** * | 50 randomly sampled items from FineTome-100k | 50 |
| **Prometheus v2.0 Evaluation** * | 50 randomly sampled items from Finance-500k | 50 |

\* [Prometheus v2.0](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) evaluations could not be completed due to Modal credit exhaustion and payment processing issues.

## Results Summary

### Key Findings

- Fine-tuning on FineTome-100k produced **marginal improvements** across test categories
- **Model size had the most significant impact** on performance
- The 3B base model consistently outperformed the 1B model regardless of fine-tuning
- FineTome's claims about producing substantial quality improvements were not validated by our tests

### Detailed Results
Complete performance breakdowns and comparative analysis are available in the `tests/reports/` directory.

## Project Structure

```
.
├── server/
│   └── src/
│       ├── llm/              # llm providers
│       ├── search/           # search the web feature
│       └── main.py           # main entrypoint for gradio server
├── test/
│   ├── test_dataset.json     # all test cases
│   ├── reports/              # detailed test results
│   ├── extract_samples.py    # helper script to sample from dataset
│   └── tests.py              # main testing entrypoint
├── ID2223_fine_tuning_.ipynb # training notebook
├── inference.py              # deployment of models script
└── README.md                 # THIS FILE

```

## Limitations

We attempted to run a comprehensive evaluation using Prometheus v2.0 on 100 randomly sampled items (50 from each dataset) but encountered Modal credit exhaustion and payment processing issues that prevented completion of this evaluation method.

---