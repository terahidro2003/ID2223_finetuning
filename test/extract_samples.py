import argparse
import json
import random
from datasets import load_dataset
import pandas as pd

def extract_samples(dataset_id, k, output_file, split="train", download_full=False):
    print(f"Loading dataset: {dataset_id} (Split: {split})...")
    
    try:
        if download_full:
            # METHOD 1: Download everything (Better randomness, high RAM/Disk usage)
            print("Mode: Full Download (Ensures perfect randomness over entire dataset)")
            ds = load_dataset(dataset_id, split=split)
            
            # Ensure we don't ask for more samples than exist
            k = min(k, len(ds))
            
            # Select random indices
            indices = random.sample(range(len(ds)), k)
            sampled_data = ds.select(indices)
            data_to_save = [item for item in sampled_data]
            
        else:
            # METHOD 2: Streaming (Instant, low RAM, approximate randomness)
            print("Mode: Streaming (Fast, low RAM usage)")
            ds = load_dataset(dataset_id, split=split, streaming=True)
            
            # Shuffle with a buffer to get randomness without downloading everything
            # buffer_size=10000 means it loads 10k items into memory and picks from them randomly
            shuffled_ds = ds.shuffle(seed=42, buffer_size=10_000)
            
            # Take K items
            data_to_save = list(shuffled_ds.take(k))

        # Determine format based on extension
        if output_file.endswith('.jsonl'):
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in data_to_save:
                    f.write(json.dumps(entry) + '\n')
        elif output_file.endswith('.csv'):
            df = pd.DataFrame(data_to_save)
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
        
        print(f"Successfully saved {len(data_to_save)} samples to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Check if the dataset requires a specific split name (e.g., 'train' vs 'main').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random samples from Hugging Face datasets.")
    
    parser.add_argument("dataset", type=str, help="Hugging Face dataset ID (e.g., 'Josephgflowers/Finance-Instruct-500k')")
    parser.add_argument("-k", "--samples", type=int, required=True, help="Number of samples to extract")
    parser.add_argument("-o", "--output", type=str, default="output.jsonl", help="Output file (supports .jsonl, .json, .csv)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--download-all", action="store_true", help="Force full download of dataset before sampling (slower, but perfectly random)")

    args = parser.parse_args()

    extract_samples(args.dataset, args.samples, args.output, args.split, args.download_all)
