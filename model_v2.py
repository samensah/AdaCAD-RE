import torch
import torch.nn as nn
import torch.nn.functional as F
import outlines
from outlines import models, generate
# from outlines.models.transformers import TransformersModel
from outlines.fsm.guide import RegexGuide
from outlines.models.tokenizer import Tokenizer
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel

# Keep your original JSD function
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



class OutlinesRelationExtractionModel(nn.Module):
    def __init__(self, model, tokenizer, max_length=30, temperature=0.1):
        super(OutlinesRelationExtractionModel, self).__init__()
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.model = model
        
        # Create a proper Outlines-compatible tokenizer
        self.outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)
        # self.outlines_model = models.Transformers(self.model, self.tokenizer)
    
    # def create_regex_guided_generator(self, pattern: str):
    #     """Create a regex-guided generator based on a pattern."""
    #     return generate.regex(self.outlines_model, pattern)
        
    
    def forward(self, original_input_ids, masked_inputs_ids, outline=None, outline_type="regex", combine_with_jsd=True):
        """Generate text with JSD-based control and optional outline constraints."""
        original_input_ids = original_input_ids.clone()
        masked_inputs_ids = masked_inputs_ids.clone()
        
        # Store original length to separate prompt from generation later
        original_length = original_input_ids.size(1)
        
        # processor to constrain model generation to follow a pattern=outline
        logits_processor = outlines.processors.RegexLogitsProcessor(regex_string=outline, tokenizer=self.outlines_tokenizer,)
        
        # Autoregressive generation for max_length steps
        for _ in range(self.max_length):
            with torch.no_grad():
                original_outputs = self.model(original_input_ids)
                masked_outputs = self.model(masked_inputs_ids)
            
            # Get logits for next token prediction (last position)
            original_logits = original_outputs.logits[:, -1, :]
            masked_logits = masked_outputs.logits[:, -1, :]

            # # Apply constraints to both distributions first
            # constrained_original_logits = logits_processor.process_logits(input_ids=original_input_ids, logits=original_logits)
            # constrained_masked_logits = logits_processor.process_logits(input_ids=masked_inputs_ids, logits=masked_logits)

            # # Calculate JSD on the constrained distributions
            # alpha = get_jsd(constrained_original_logits, constrained_masked_logits)

            # adjusted_logits=constrained_original_logits
            # # Adjust logits based on divergence within the constrained space
            # if combine_with_jsd:
            #     # adjusted_logits = (1 + alpha) * constrained_original_logits - alpha * constrained_masked_logits
            #     adjusted_logits = (1 + alpha) * constrained_masked_logits - alpha * constrained_original_logits

            # Calculate JSD
            alpha = get_jsd(original_logits, masked_logits)
            
            adjusted_logits=original_logits
            # Adjust logits based on divergence
            if combine_with_jsd:
                adjusted_logits = (1 + alpha) * (original_logits) - alpha * (masked_logits)
            
            # Apply the guide to constrain logits 
            adjusted_logits = logits_processor.process_logits(input_ids=original_input_ids, logits=adjusted_logits)

            # Convert to probs with temperature scaling
            adjusted_probs = F.softmax(adjusted_logits / self.temperature, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(adjusted_probs, num_samples=1)

            # Add to sequence
            original_input_ids = torch.cat([original_input_ids, next_token], dim=-1)
            
            # # End generation if we reached an EOS token
            # if next_token.item() == self.tokenizer.eos_token_id:
            #     break

        generated_tokens = original_input_ids[:, original_length:]
        # Decode generated tokens to text
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return decoded_outputs

# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    
    # Initialize the model with Outlines support
    model = OutlinesRelationExtractionModel(
        base_model, 
        tokenizer,
        max_length=30,
        temperature=0.1
    )
    
    # Create dataset and dataloader
    dataset = RelationExtractionDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda")
    )

    regex_pattern = r"(no_relation|pers:title:title|org:gpe:operations_in|pers:org:employee_of|org:org:agreement_with|org:date:formed_on|pers:org:member_of|org:org:subsidiary_of|org:org:shares_of|org:money:revenue_of|org:money:loss_of|org:gpe:headquartered_in|org:date:acquired_on|pers:org:founder_of|org:gpe:formed_in|org:org:acquired_by|pers:univ:employee_of|pers:gov_agy:member_of|pers:univ:attended|pers:univ:member_of|org:money:profit_of|org:money:cost_of)"
    
    # Run inference
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():  # Disable gradient computation for inference
            # Process the batch through our model
            outputs = model(
                batch['original_inputs']['input_ids'],  # Original text
                batch['masked_inputs']['input_ids'],     # Text with masked entities
                outline=regex_pattern,
                outline_type="regex",
                combine_with_jsd=True  # Use custom integration
            )
            
            # Display results
            print("\nGenerated relation classifications:")
            for i, output in enumerate(outputs):
                print(f"Predicted relation: {output.strip()}")
                print(f"Actual relation: {batch['relations'][i]}")
                print("-" * 50)
                
            
        # Only process the first batch for demonstration
        break




    # # Example prompt
    # prompt_ids = tokenizer.encode("John Smith (a person) has no relationship with Microsoft.", return_tensors="pt")
    # masked_ids = tokenizer.encode("SUBJECT-PERSON is an employee of OBJECT-ORG.", return_tensors="pt")
    
    
    # # Example 3: Combining JSD with Outlines constraints
    # # This uses our custom integration to bias tokens with BOTH methods
    # combined_output = model(
    #     prompt_ids,
    #     masked_ids,
    #     outline=regex_pattern,
    #     outline_type="regex",
    #     combine_with_jsd=True  # Use custom integration
    # )
    # print("Combined output (JSD + Outlines):", combined_output)
    
    # # Example 4: Using JSON schema (Outlines only, no JSD)
    # json_output = model(
    #     prompt_ids, 
    #     masked_ids, 
    #     outline=json_schema, 
    #     outline_type="json_schema",
    #     combine_with_jsd=False  # JSON schema requires using Outlines' generator
    # )
    # print("JSON schema-constrained output:", json_output)