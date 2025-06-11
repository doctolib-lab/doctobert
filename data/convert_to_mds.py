from datasets import load_dataset
from streaming import MDSWriter
import argparse
import os

def convert_to_mds(dataset, output_path):
    with MDSWriter(out=output_path, columns={"text": "str"}) as writer:
        for record in dataset:
            writer.write(record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/test MDS files")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset for test split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)")
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_dataset("text", data_files=args.input_path, split="train", streaming=False)
    
    # Create train/test split
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert train split to MDS
    train_output_path = os.path.join(args.output_dir, "train")
    print(f"Converting train set ({len(split_dataset['train'])} samples) to MDS format...")
    convert_to_mds(split_dataset['train'], train_output_path)
    
    # Convert test split to MDS
    val_output_path = os.path.join(args.output_dir, "val")
    print(f"Converting val set ({len(split_dataset['test'])} samples) to MDS format...")
    convert_to_mds(split_dataset['test'], val_output_path)
    
    print("Train/val split completed!")
    print(f"Train set saved to: {train_output_path}")
    print(f"Val set saved to: {val_output_path}")
