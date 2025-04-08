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


class CustomGuideLogitsProcessor:
    """A custom implementation of GuideLogitsProcessor based on the Outlines code"""
    
    def __init__(self, tokenizer, guide):
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None
        
    def process_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.Tensor:
        """Use the Guide to bias the logits before sampling the next token."""
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx:]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        allowed_tokens_batch = []
        batch_indices = []
        for i, guide_state in enumerate(sequence_states):
            allowed_tokens = self.guide.get_next_instruction(guide_state).tokens
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))  # Store batch index for each allowed token

        allowed_tokens_concat = torch.cat(allowed_tokens_batch).to(logits.device)
        batch_indices_concat = torch.cat(batch_indices).to(logits.device)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[batch_indices_concat, allowed_tokens_concat] = False
        logits.masked_fill_(mask, float("-inf"))

        return logits


class OutlinesRelationExtractionModel(nn.Module):
    def __init__(self, model, tokenizer, max_length=30, temperature=0.1):
        super(OutlinesRelationExtractionModel, self).__init__()
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.model = model
        
        # Create a proper Outlines-compatible tokenizer
        self.outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)
        self.outlines_model = models.Transformers(self.model, self.tokenizer)
    
    def create_regex_guided_generator(self, pattern: str):
        """Create a regex-guided generator based on a pattern."""
        return generate.regex(self.outlines_model, pattern)
        
    def create_regex_guide(self, regex_string: str):
        """Create a RegexGuide directly (lower level than the generator)."""
        return RegexGuide.from_regex(regex_string, self.outlines_tokenizer)
    
    def forward(self, original_input_ids, masked_inputs_ids, outline=None, outline_type="regex", combine_with_jsd=True):
        """
        Generate text with JSD-based control and optional outline constraints.
        
        Args:
            original_input_ids: Input IDs for the original text
            masked_inputs_ids: Input IDs for the masked text
            outline: Constraint for generation (regex pattern or JSON schema)
            outline_type: Type of outline ('regex' or 'json_schema')
            combine_with_jsd: If True, combine outline constraints with JSD biasing
        """
        original_input_ids = original_input_ids.clone()
        masked_inputs_ids = masked_inputs_ids.clone()
        
        # Store original length to separate prompt from generation later
        original_length = original_input_ids.size(1)
        
        # Convert input_ids to prompt text for Outlines
        prompt = self.tokenizer.decode(original_input_ids[0])
        
        # Create a guide if we're using constraints with JSD
        guide = None
        if outline is not None and combine_with_jsd:
            guide = self.create_regex_guide(outline)
            
            # Create our logits processor with the guide
            logits_processor = CustomGuideLogitsProcessor(self.outlines_tokenizer, guide)
        
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
            
            # Adjust logits based on divergence
            adjusted_logits = (1 + alpha) * (original_logits) - alpha * (masked_logits)
            
            # Apply the guide to constrain logits if needed
            if guide is not None:
                adjusted_logits = logits_processor.process_logits(original_input_ids, adjusted_logits)

            # Convert to probs with temperature scaling
            adjusted_probs = F.softmax(adjusted_logits / self.temperature, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(adjusted_probs, num_samples=1)

            # Add to sequence
            original_input_ids = torch.cat([original_input_ids, next_token], dim=-1)
            
            # End generation if we reached an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_tokens = original_input_ids[:, original_length:]
            # Decode generated tokens to text
            decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(decoded_outputs)
        
        return decoded_outputs

# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use configuration instead of hardcoded values
    base_model_name = "microsoft/Phi-4-mini-instruct"  # Example model
    
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
    
    # Example: Using regex pattern to force output to be a relation type
    regex_pattern = r"(no_relation|pers:title:title|org:gpe:operations_in|pers:org:employee_of|org:org:agreement_with|org:date:formed_on|pers:org:member_of|org:org:subsidiary_of|org:org:shares_of|org:money:revenue_of|org:money:loss_of|org:gpe:headquartered_in|org:date:acquired_on|pers:org:founder_of|org:gpe:formed_in|org:org:acquired_by|pers:univ:employee_of|pers:gov_agy:member_of|pers:univ:attended|pers:univ:member_of|org:money:profit_of|org:money:cost_of)"
    
    # # Example: Using JSON schema to structure the output
    # json_schema = {
    #     "type": "object",
    #     "properties": {
    #         "relation": {
    #             "type": "string",
    #             "enum": ["is_parent_of", "is_child_of", "is_spouse_of", "is_sibling_of"]
    #         },
    #         "confidence": {
    #             "type": "number",
    #             "minimum": 0,
    #             "maximum": 1
    #         }
    #     },
    #     "required": ["relation", "confidence"]
    # }
    
    # Example prompt
    prompt_ids = tokenizer.encode("Google announced that John Smith will join Microsoft as the new chief marketing officer.", return_tensors="pt")
    masked_ids = tokenizer.encode("Google announced that SUBJECT-PERSON will join OBJECT-ORG as the new chief marketing officer.", return_tensors="pt")
    
    # # Example 1: Using Outlines' built-in generators (high-level API)
    # regex_output = model(
    #     prompt_ids, 
    #     masked_ids, 
    #     outline=regex_pattern, 
    #     outline_type="regex",
    #     combine_with_jsd=False  # Use Outlines' generator directly
    # )
    # print("Regex-constrained output (Outlines only):", regex_output)
    
    # # Example 2: Using JSD-driven generation without constraints
    # unconstrained_output = model(
    #     prompt_ids,
    #     masked_ids,
    #     outline=None  # No constraints
    # )
    # print("Unconstrained output (JSD only):", unconstrained_output)
    
    # Example 3: Combining JSD with Outlines constraints
    # This uses our custom integration to bias tokens with BOTH methods
    combined_output = model(
        prompt_ids,
        masked_ids,
        outline=regex_pattern,
        outline_type="regex",
        combine_with_jsd=True  # Use custom integration
    )
    print("Combined output (JSD + Outlines):", combined_output)
    
    # # Example 4: Using JSON schema (Outlines only, no JSD)
    # json_output = model(
    #     prompt_ids, 
    #     masked_ids, 
    #     outline=json_schema, 
    #     outline_type="json_schema",
    #     combine_with_jsd=False  # JSON schema requires using Outlines' generator
    # )
    # print("JSON schema-constrained output:", json_output)