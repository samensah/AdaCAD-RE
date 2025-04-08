import torch
import logging
import os
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from config import config
from model import RelationExtractionModel
from dataset import RelationExtractionDataset
from utils import custom_collate_fn

def setup_logging():
    """Set up logging configuration"""
    log_level = getattr(logging, config.log_level.upper() if hasattr(config, "log_level") else "INFO")
    
    # Create output directory if it doesn't exist
    if hasattr(config, "output_dir") and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Extract model name and dataset path for log filename
    model_name = os.path.basename(config.base_model_name)
    dataset_name = config.data_path.replace("/", "_").replace(".", "_")
    log_filename = f"{model_name}_{dataset_name}.log"
    
    # Setup basic logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # Simple format for relations
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting relation extraction with config: {vars(config)}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading base model from {config.base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Move model to specified device
    device = torch.device(config.device)
    base_model = base_model.to(device)
    logger.info(f"Using device: {device}")
    
    # Set model to evaluation mode
    base_model.eval()
    
    # Initialize relation extraction model
    logger.info("Initializing relation extraction model")
    model = RelationExtractionModel(
        base_model, 
        tokenizer,
        max_length=config.max_length,
        temperature=config.temperature
    )
    
    # Create dataset
    logger.info(f"Loading dataset from {config.data_path}")
    dataset = RelationExtractionDataset(config.data_path, tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.device
    )
    
    # Run model evaluation
    logger.info("Starting model evaluation")
    results = []
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        
        # Move inputs to device
        for key in batch["original_inputs"]:
            batch["original_inputs"][key] = batch["original_inputs"][key].to(device)
        for key in batch["masked_inputs"]:
            batch["masked_inputs"][key] = batch["masked_inputs"][key].to(device)
        
        with torch.no_grad():  # Disable gradient computation for inference
            # Process the batch through our model
            outputs = model(
                batch['original_inputs']['input_ids'],
                batch['masked_inputs']['input_ids']
            )
            
            # Compare predictions with ground truth
            for i, output in enumerate(outputs):
                predicted_relation = output.strip()
                actual_relation = batch['relations'][i]
                
                # # Log results
                # logger.info(f"Sample {total+i+1}:")
                # logger.info(f"  Predicted: {predicted_relation}")
                # logger.info(f"  Actual: {actual_relation}")
                
                # Check if prediction is correct
                is_correct = predicted_relation == actual_relation
                if is_correct:
                    correct += 1
                
                # Store result
                results.append({
                    "id": batch["instance_id"][i].item(),
                    "original_text": batch["original_texts"][i],
                    "predicted": predicted_relation,
                    "actual": actual_relation,
                    "correct": is_correct
                })
                
                
        # Update total count
        total += len(outputs)
        
        # Calculate and log accuracy
        accuracy = correct / total
        logger.info(f"Accuracy so far: {accuracy:.4f} ({correct}/{total})")
    
    # Final accuracy
    final_accuracy = correct / total
    logger.info(f"Final accuracy: {final_accuracy:.4f} ({correct}/{total})")
    
    # Save results
    import json
    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": final_accuracy,
            "samples": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()