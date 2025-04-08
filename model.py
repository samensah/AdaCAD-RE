import torch
import torch.nn as nn
import accelerate
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import torch.nn.functional as F

def get_jsd(p, q):
    """Calculate Jensen-Shannon Divergence between two distributions."""
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
     # Calculate the midpoint distribution m = (p+q)/2
    if ((p + q) == 0).any():
        m = (0.5 * (p + q)).clamp_min(1e-9).log()
    else:
        m = (0.5 * (p + q)).log()
    # Ensure no negative or zero probabilities to avoid numerical instability
    if torch.any(p <= 0):
        p = p.clamp_min(1e-9)
    if torch.any(q <= 0):
        q = q.clamp_min(1e-9)
    # JSD is the average of KL divergences from p to m and q to m
    return 0.5 * (F.kl_div(m, p, reduction='batchmean', log_target=False) + 
                  F.kl_div(m, q, reduction='batchmean', log_target=False))



class RelationExtractionModel(nn.Module):
    
    def __init__(self, model, tokenizer, max_length=30, temperature=0.1):
        super(RelationExtractionModel, self).__init__()
        self.max_length = max_length
        self.temperature =temperature
        self.tokenizer = tokenizer
        self.model = model

    
    
    def forward(self, original_input_ids, masked_inputs_ids):

        original_input_ids = original_input_ids.clone()
        masked_inputs_ids = masked_inputs_ids.clone()
        # Store original length to separate prompt from generation later
        original_length = original_input_ids.size(1)

        # Autoregressive generation for max_length steps
        for _ in range(self.max_length):
            # Run inputs through the model
            with torch.no_grad():
                original_outputs = self.model(original_input_ids)
                masked_outputs = self.model(masked_inputs_ids)
            
             # Get logits for next token prediction (last position)
            original_logits = original_outputs.logits[:, -1, :]
            masked_logits = masked_outputs.logits[:, -1, :]

            # Calculate JSD
            alpha = get_jsd(original_logits, masked_logits)
            # Adjust logits based on divergence:
            # - When alpha is high (distributions differ), favor original_logits more
            # - When alpha is low (distributions similar), slightly favor original_logits
            adjusted_logits = (1 + alpha) * (original_logits) - alpha * (masked_logits)

            # convert to probs with temperature scaling
            adjusted_probs = F.softmax(adjusted_logits / self.temperature, dim=-1)
            # sample next token
            next_token = torch.multinomial(adjusted_probs, num_samples=1)

            original_input_ids = torch.cat([original_input_ids, next_token], dim=-1)

            # if next_token.item() == tokenizer.eos_token_id:
            #     break

        generated_tokens = original_input_ids[:, original_length:]
    
        # Decode generated tokens to text
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return decoded_outputs



# Example usage
if __name__ == "__main__":
    from dataset import RelationExtractionDataset
    from constant import refind_relation_descriptions
    from utils import custom_collate_fn
    from torch.utils.data import Dataset, DataLoader

    from dataset import RelationExtractionDataset
    from utils import custom_collate_fn
    from torch.utils.data import Dataset, DataLoader
    from config import config
    
    # Use configuration instead of hardcoded values
    base_model_name = config.base_model_name
    data_path = config.data_path
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()
    print(f"Base model embeddings resized to accommodate {len(tokenizer)} tokens")

    # Initialize relation extraction model with config values
    model = RelationExtractionModel(
        base_model, 
        tokenizer,
        max_length=config.max_length,
        temperature=config.temperature
    )

    # Create dataset and dataloader
    dataset = RelationExtractionDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda")
    )

    # Run inference
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():  # Disable gradient computation for inference
            # Process the batch through our model
            outputs = model(
                batch['original_inputs']['input_ids'],  # Original text
                batch['masked_inputs']['input_ids']     # Text with masked entities
            )
            
            # Display results
            print("\nGenerated relation classifications:")
            for i, output in enumerate(outputs):
                print(f"Sample {i+1}: {output.strip()}")
                print(f"Actual relation: {batch['relation'][i]}")
                print("-" * 50)
                
            
        # Only process the first batch for demonstration
        break

    