import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from utils import custom_collate_fn
from tqdm import tqdm
from constant import refind_relation_descriptions
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RelationExtractionDataset(Dataset):
    """Dataset for relation extraction task."""
    
    # Class variables to store train data embeddings
    train_data = None
    train_embeddings = None
    sentence_model = None
    train_relation_to_instances = None
    device = "cpu"  # Default device
    
    @classmethod
    def prepare_train_examples(cls, train_path, num_examples=3, device="cpu"):
        """Prepare training data examples and compute embeddings once."""
        # Store device for later use
        cls.device = device
        # Load training data

        with open(train_path, 'r') as f:
            cls.train_data = json.load(f)
            
        # Group instances by relation for efficient retrieval
        cls.train_relation_to_instances = {}
        for idx, instance in enumerate(cls.train_data):
            relation = instance["relation"]
            if relation not in cls.train_relation_to_instances:
                cls.train_relation_to_instances[relation] = []
            cls.train_relation_to_instances[relation].append((idx, instance))
            
        # Initialize sentence transformer if not already done
        if cls.sentence_model is None:
            cls.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # Compute embeddings for train data
        print(f"Computing sentence embeddings for {len(cls.train_data)} training instances...")
        train_texts = [" ".join(instance["token"]) for instance in cls.train_data]
        cls.train_embeddings = cls.sentence_model.encode(train_texts, show_progress_bar=True, convert_to_tensor=True, device=device)
        
        print(f"Prepared {len(cls.train_data)} training instances for examples")
        
    def __init__(self, data_path, tokenizer, data_description_func, num_examples=3, device="cpu"):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data = self._load_data(data_path)
        self.data_description_func = data_description_func
        self.num_examples = num_examples
        self.device = device

        # Set class device if not already set
        if self.__class__.device != device:
            self.__class__.device = device
        
        # Pre-compute unique relations
        self.unique_relations = sorted(list(set([item["relation"] for item in self.data])))
        print(f"Loaded {len(self.data)} instances with {len(self.unique_relations)} relation types")
    
    def _load_data(self, data_path):
        """Load data from JSON file."""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def _find_similar_examples(self, instance):
        """Find similar examples from the training set with the same relation."""
        # Skip if train data not prepared
        if RelationExtractionDataset.train_data is None:
            return []
            
        relation = instance["relation"]
        instance_text = " ".join(instance["token"])
        
        # Compute embedding for current instance
        instance_embedding = RelationExtractionDataset.sentence_model.encode(
            instance_text, convert_to_tensor=True,
            device=self.__class__.device  # Use the class device
        )
            
        # Get all training instances with the same relation
        same_relation_instances = RelationExtractionDataset.train_relation_to_instances.get(relation, [])
        
        # If not enough examples with same relation, return available ones
        if len(same_relation_instances) <= self.num_examples:
            return [inst for _, inst in same_relation_instances][:self.num_examples]
        
        # Get embeddings for training instances with the same relation
        indices = [idx for idx, _ in same_relation_instances]
        candidate_embeddings = RelationExtractionDataset.train_embeddings[indices]
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(
            instance_embedding.cpu().numpy().reshape(1, -1), 
            candidate_embeddings.cpu().numpy()
        )[0]
        
        # Get top similar instances
        most_similar_indices = np.argsort(similarity_scores)[::-1][:self.num_examples]
        
        # Return the examples
        return [same_relation_instances[idx][1] for idx in most_similar_indices]
        
    def _create_masked_instance(self, instance):
        """Create a masked version of the instance."""
        masked_instance = instance.copy()
        tokens = instance["token"].copy()
        # Get entity spans
        subj_start, subj_end = instance["subj_start"], instance["subj_end"]
        obj_start, obj_end = instance["obj_start"], instance["obj_end"]
        
        masked_tokens = tokens.copy()
        
        # Replace object first (in case subject and object overlap)
        obj_placeholder = f"OBJECT-{instance['obj_type']}"
        masked_tokens[obj_start:obj_end+1] = [obj_placeholder]
        
        # Adjust subject indices if object was before subject and changed token count
        if obj_end < subj_start:
            obj_token_count = obj_end - obj_start + 1 # num of tokens removed
            subj_start -= (obj_token_count - 1)
            subj_end -= (obj_token_count - 1)
        
        # Replace subject
        subj_placeholder = f"SUBJECT-{instance['subj_type']}"
        masked_tokens[subj_start:subj_end+1] = [subj_placeholder]
        
        masked_instance["token"] = masked_tokens
        return masked_instance
    
    def _get_filtered_relation_descriptions(self, instance, masked=False):
        """Get relation descriptions filtered by entity types for this instance"""
        # Get entity texts based on masked parameter
        if masked:
            subj_text = f"SUBJECT-{instance['subj_type']}"
            obj_text = f"OBJECT-{instance['obj_type']}"
        else:
            subj_text = " ".join(instance["token"][instance["subj_start"]:instance["subj_end"]+1])
            obj_text = " ".join(instance["token"][instance["obj_start"]:instance["obj_end"]+1])
        
        # Map entity types to relation format (PERSON -> pers)
        def map_entity_type(entity_type):
            if entity_type == "PERSON":
                return "pers"
            return entity_type.lower()
        
        # Get the subject and object prefixes for this instance
        subj_prefix = map_entity_type(instance['subj_type'])
        obj_prefix = map_entity_type(instance['obj_type'])
        
        # Generate descriptions with the appropriate entity text
        all_relations = self.data_description_func(subj_text, obj_text)
        if not self.data_path.startswith('data/refind'):
            return all_relations
        
        # Always include no_relation
        filtered_relations = {"no_relation": all_relations["no_relation"]}

        
        # Filter relations based on entity types
        for relation, description in all_relations.items():
            if relation == "no_relation":
                continue  # Already added
            
            # Parse the relation format (e.g., "pers:org:employee_of")
            parts = relation.split(":")
            relation_subj, relation_obj = parts[0], parts[1]
            
            # Include only if the subject and object types match the instance's entity types
            if relation_subj == subj_prefix and relation_obj == obj_prefix:
                filtered_relations[relation] = description
        
        return filtered_relations
    
    def _format_example(self, example, masked=False):
        """Format an example instance for inclusion in the prompt."""
        example_to_use = self._create_masked_instance(example) if masked else example
        
        # Extract entity text
        subj_tokens = " ".join(example["token"][example["subj_start"]:example["subj_end"]+1])
        obj_tokens = " ".join(example["token"][example["obj_start"]:example["obj_end"]+1])
        
        # Format the example
        if masked:
            text = f"Text: {' '.join(example_to_use['token'])}\n"
            text += f"The relation between SUBJECT-{example['subj_type']} and OBJECT-{example['obj_type']} is: {example['relation']}\n"
        else:
            text = f"Text: {' '.join(example_to_use['token'])}\n"
            text += f"Subject Entity: {subj_tokens} (Type: {example['subj_type']})\n"
            text += f"Object Entity: {obj_tokens} (Type: {example['obj_type']})\n"
            text += f"The relation between the subject {subj_tokens} and object {obj_tokens} is: {example['relation']}\n"
        
        return text
    
    def _format_instance(self, instance, masked=False):
        """Format instance as a prompt for the model."""
        instance_to_use = self._create_masked_instance(instance) if masked else instance
        tokens = instance_to_use["token"]
        
        # Create a nicely formatted prompt
        formatted_text = " ".join(tokens)
        
        # Add a prompt asking for the relation
        prompt = f"Text: {formatted_text}\n\nGiven the {{Text}}, what is the relation between "
        
        # Extract entity text
        subj_tokens = " ".join(instance["token"][instance["subj_start"]:instance["subj_end"]+1])
        obj_tokens = " ".join(instance["token"][instance["obj_start"]:instance["obj_end"]+1])
        
        if masked:
            prompt += f"SUBJECT-{instance['subj_type']} and OBJECT-{instance['obj_type']}?\n\n"
        else:
            prompt += f"'{subj_tokens}' and '{obj_tokens}'?\n\n"
            prompt += f"Subject Entity: {subj_tokens}  (Type: {instance['subj_type']})\n"
            prompt += f"Object Entity: {obj_tokens}  (Type: {instance['obj_type']})\n"
        
        prompt += "Choose only one from the following relations:\n\n"
        
        # Get relation descriptions filtered for this instance - more efficient!
        # Pass the masked parameter to use appropriate entity text in the descriptions
        filtered_relations = self._get_filtered_relation_descriptions(instance, masked=masked)
        
        # Add enumerated relation descriptions
        for i, rel_name in enumerate(filtered_relations):
            prompt += f"{i}. {rel_name}: {filtered_relations[rel_name]}\n"
        
        # Add examples if the training data is prepared
        if self.num_examples > 0 and RelationExtractionDataset.train_data is not None:
            # Find similar examples with the same relation
            similar_examples = self._find_similar_examples(instance)
            
            if similar_examples:
                prompt += "\nHere are some examples:\n\n"
                for example in similar_examples:
                    prompt += self._format_example(example, masked=masked)
                    prompt += "\n"

        # Add the final instruction
        if masked:
            prompt += f"The relation between SUBJECT-{instance['subj_type']} and OBJECT-{instance['obj_type']} is:"
        else:
            prompt += f"The relation between the subject {subj_tokens} and object {obj_tokens} is:"
        
        return prompt, list(filtered_relations.keys())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        instance = self.data[idx]
        
        # Create both original and masked prompts with the relation names
        original_prompt, relation_names = self._format_instance(instance, masked=False)
        masked_prompt, _ = self._format_instance(instance, masked=True)
        
        # Tokenize prompts
        original_inputs = self.tokenizer(original_prompt,  return_tensors="pt") # max_length=512, truncation=True,
        masked_inputs = self.tokenizer(masked_prompt,  truncation=True, return_tensors="pt")

        
        # Remove batch dimension
        for key in original_inputs:
            original_inputs[key] = original_inputs[key].squeeze(0)
        for key in masked_inputs:
            masked_inputs[key] = masked_inputs[key].squeeze(0)

        #  Extract entity texts
        subj_text = " ".join(instance["token"][instance["subj_start"]:instance["subj_end"]+1])
        obj_text = " ".join(instance["token"][instance["obj_start"]:instance["obj_end"]+1])
        
        return {
            "instance_id": idx,
            "original_inputs": original_inputs,
            "masked_inputs": masked_inputs,
            "relation": instance["relation"],
            "original_text": original_prompt,
            "masked_text": masked_prompt,
            "subject_entity": subj_text,
            "object_entity": obj_text,
            "subject_type": instance["subj_type"],
            "object_type": instance["obj_type"]
        }

if __name__ == "__main__":
    from config import config
    # Data paths
    # base_model_name = "microsoft/Phi-4-mini-instruct"
    base_model_name = config.base_model_name
    train_path = f"{config.data_path}/train.json"
    test_path = f"{config.data_path}/test.json"
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare training data examples first (computes embeddings once)
    RelationExtractionDataset.prepare_train_examples(train_path, num_examples=3)
    
    # Create test and train datasets (they'll share the same train examples)
    test_dataset = RelationExtractionDataset(
        test_path, 
        tokenizer, 
        data_description_func=refind_relation_descriptions,
        num_examples=3
    )
    
    # Create dataloaders
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda")
    )
    
    print(f"Test dataset created with {len(test_dataset)} samples")


