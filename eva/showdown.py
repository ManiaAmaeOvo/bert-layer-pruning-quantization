# ==============================================================================
# Final Showdown: Comprehensive Model Evaluation Script (Final Corrected)
# ==============================================================================
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import numpy as np
import os
import pandas as pd
from typing import Optional, Tuple, Union

# --- 1. Configuration ---
print("--- 1. Configuring The Final Showdown ---")
MODELS_TO_EVALUATE = [
    {"name": "1. FP32 Baseline (12L)", "type": "fp32", "path": "./my_bert_sst2_finetuned/checkpoint-1800"},
    {"name": "2. INT4 BitsAndBytes (12L, GPU-Only)", "type": "int4_bnb", "path": "./models/bert_sst2_int4_bnb"},
    {"name": "3. INT8 PTQ (12L, CPU-Only)", "type": "int8_ptq", "path": "./models/bert_sst2_int8_ptq.pt"},
    {"name": "4. INT8 QAT (12L)", "type": "fp32", "path": "./models/bert_sst2_int8_qat_optimum"},
    {"name": "5. Pruned FP32 (8L)", "type": "fp32", "path": "./models/bert_pruned_8_layers_finetuned/best_model"},
    {"name": "6. Pruned FP16 (8L, GPU-Only)", "type": "fp16_gpu", "path": "./models/bert_pruned_fp16_gpu"},
]
TOKENIZER_NAME = "bert-base-uncased"
TEST_SENTENCE = "This movie is not bad at all, in fact it is surprisingly good!"
LATENCY_RUNS = 100
EVAL_BATCH_SIZE = 32

# --- 2. Helper Functions ---
print("--- 2. Defining Helper Functions ---")

def get_model_size(path, model_type):
    """Calculates the size of a model on disk."""
    if model_type == 'int8_ptq':
        return os.path.getsize(path) / (1024 * 1024)
    else:
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)

def load_model_and_tokenizer(config, device):
    """Loads a model and tokenizer based on the provided configuration."""
    # The fix is to import torch inside the function to avoid scope issues.
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model_type = config['type']
    path = config['path']
    model = None

    if model_type == 'fp32':
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
    elif model_type == 'int4_bnb':
        if device.type != 'cuda': raise ValueError("BitsAndBytes INT4 model requires GPU.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, quantization_config=quantization_config, device_map="auto")
    elif model_type == 'int8_ptq':
        if device.type != 'cpu': raise ValueError("PyTorch Dynamic PTQ INT8 model is designed for CPU.")
        model_fp32_skeleton = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)
        import torch.quantization
        model_int8_skeleton = torch.quantization.quantize_dynamic(model_fp32_skeleton, {torch.nn.Linear}, dtype=torch.qint8)
        model_int8_skeleton.load_state_dict(torch.load(path, map_location='cpu'))
        model = model_int8_skeleton
        model.to(device)
    elif model_type == 'fp16_gpu':
        if device.type != 'cuda': raise ValueError("FP16 model requires GPU.")
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16)
        model.to(device)

    if model is None: raise ValueError(f"Unknown model type: {model_type}")
    model.eval()
    return model, tokenizer

def measure_latency(model, tokenizer, device, model_type):
    """Measures the average inference latency of a model."""
    target_device = device
    if 'bnb' in model_type: target_device = next(model.parameters()).device
    inputs = tokenizer(TEST_SENTENCE, return_tensors="pt").to(target_device)
    timings = []
    with torch.no_grad():
        for _ in range(20):
            _ = model(**inputs)
        for _ in range(LATENCY_RUNS):
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(**inputs)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
    return np.mean(timings) * 1000

def evaluate_accuracy(model, tokenizer, device, eval_dataloader, model_type):
    """Evaluates the accuracy of a model on the validation set."""
    metric = evaluate.load("accuracy")
    for batch in tqdm(eval_dataloader, desc=f"Evaluating Acc on {device.type.upper()}", leave=False):
        target_device = device
        if 'bnb' in model_type: target_device = next(model.parameters()).device
        label_key = "labels" if "labels" in batch else "label"
        batch_on_device = {'input_ids': batch['input_ids'].to(target_device), 'attention_mask': batch['attention_mask'].to(target_device)}
        with torch.no_grad():
            outputs = model(**batch_on_device)
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch[label_key])
    return metric.compute()["accuracy"]

# --- 3. Prepare Dataset ---
print("--- 3. Preparing Dataset ---")
raw_datasets = load_dataset("glue", "sst2")
tokenizer_for_map = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
def tokenize_function(examples):
    return tokenizer_for_map(examples["sentence"], padding="max_length", truncation=True, max_length=128)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets.set_format('torch')
eval_dataset = tokenized_datasets["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE)

# --- 4. Run Main Evaluation Loop ---
results = []
for config in MODELS_TO_EVALUATE:
    print(f"\n{'='*20} EVALUATING: {config['name']} {'='*20}")
    current_results = {"Model": config['name']}
    current_results["Size (MB)"] = get_model_size(config['path'], config['type'])
    
    if torch.cuda.is_available():
        if config['type'] in ['int8_ptq']:
            print(f"--- SKIPPING GPU Evaluations for {config['name']} (CPU-only) ---")
            current_results["Peak GPU Mem (MB)"] = "N/A"
            current_results["Latency (GPU, ms)"] = "N/A"
            current_results["Accuracy (GPU)"] = "N/A"
        else:
            device = torch.device("cuda")
            print(f"--- GPU Evaluations ({config['name']}) ---")
            torch.cuda.reset_peak_memory_stats(device)
            model, tokenizer = load_model_and_tokenizer(config, device)
            
            target_device_mem = model.device if not hasattr(model, 'device_map') else next(model.parameters()).device
            inputs_for_mem = tokenizer(TEST_SENTENCE, return_tensors="pt").to(target_device_mem)
            with torch.no_grad():
                _ = model(**inputs_for_mem)
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            current_results["Peak GPU Mem (MB)"] = peak_mem_mb
            
            current_results["Latency (GPU, ms)"] = measure_latency(model, tokenizer, device, config['type'])
            current_results["Accuracy (GPU)"] = evaluate_accuracy(model, tokenizer, device, eval_dataloader, config['type'])
            
            del model
            torch.cuda.empty_cache()
            
    if 'bnb' in config['type'] or 'fp16_gpu' in config['type']:
        print(f"--- SKIPPING CPU Evaluations for {config['name']} (GPU-only) ---")
        current_results["Latency (CPU, ms)"] = "N/A"
        current_results["Accuracy (CPU)"] = "N/A"
    else:
        device = torch.device("cpu")
        print(f"--- CPU Evaluations ({config['name']}) ---")
        model, tokenizer = load_model_and_tokenizer(config, device)
        current_results["Latency (CPU, ms)"] = measure_latency(model, tokenizer, device, config['type'])
        current_results["Accuracy (CPU)"] = evaluate_accuracy(model, tokenizer, device, eval_dataloader, config['type'])
        del model
        
    results.append(current_results)

# --- 5. Display and Save Final Results Table ---
print(f"\n{'='*25} FINAL RESULTS {'='*25}")
df = pd.DataFrame(results)

df["Size (MB)"] = df["Size (MB)"].map('{:.2f}'.format)
if "Peak GPU Mem (MB)" in df.columns:
    df["Peak GPU Mem (MB)"] = df["Peak GPU Mem (MB)"].apply(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)
    df["Latency (GPU, ms)"] = df["Latency (GPU, ms)"].apply(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)
    df["Accuracy (GPU)"] = df["Accuracy (GPU)"].apply(lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x)
df["Latency (CPU, ms)"] = df["Latency (CPU, ms)"].apply(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)
df["Accuracy (CPU)"] = df["Accuracy (CPU)"].apply(lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x)

if torch.cuda.is_available():
    column_order = [
        "Model", "Size (MB)", 
        "Accuracy (GPU)", "Latency (GPU, ms)", "Peak GPU Mem (MB)", 
        "Accuracy (CPU)", "Latency (CPU, ms)"
    ]
    for col in column_order:
        if col not in df.columns: df[col] = 'N/A'
else:
    column_order = ["Model", "Size (MB)", "Accuracy (CPU)", "Latency (CPU, ms)"]
df = df[column_order]

markdown_string = df.to_markdown(index=False)
print(markdown_string)
output_filename = "final_results.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_string)
print(f"\n✅ 结果已成功保存到文件: {output_filename}")

