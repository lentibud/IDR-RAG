import json
import requests
import argparse
from datasets import load_from_disk
import numpy as np
from Pipeline import RAGPipeline


def normalize_text(text):
    return text.lower().strip()


def evaluate_single_hotpotqa(dataset_path, idx, model_name="llama2:13b"):
    """
    Evaluate a single sample from the HotpotQA dataset.

    Args:
        dataset_path (str): Path to the HotpotQA dataset directory.
        idx (int): Index of the sample to evaluate within the dataset.
        model_name (str): Name of the model to use for generating the answer.
    """
    pipeline = RAGPipeline()
    dataset = load_from_disk(dataset_path)

    # Check if the provided index is within valid range
    if idx < 0 or idx >= len(dataset):
        print(f"Error: Index {idx} is out of range. The dataset contains {len(dataset)} samples.")
        return

    example = dataset[int(idx)]
    
    question = example['question']
    ground_truth = example['answer']
    
    # Generate answer using the RAG pipeline
    answer = pipeline.generate_answer(question, example, model_name).strip().lower()
    
    print("===== Result =====")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Model Answer: {answer}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Evaluate a single sample by index.")

    # The idx argument
    parser.add_argument(
        '--idx', 
        type=int, 
        required=True,
        help='Index of the sample to evaluate'
    )

    # Parse arguments
    args = parser.parse_args()

    # Paths and model name
    dataset_path = "./hotpotqa_validation"
    model_name = "llama2:13b"

    # Call evaluation
    evaluate_single_hotpotqa(
        dataset_path=dataset_path,
        idx=args.idx,
        model_name=model_name
    )