import torch 
import os
import random
import copy
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def custom_collate_fn(batch):
    # Initialize the batch dictionary to store batched tensors
    batch_dict = {
        "instance_id": [],
        "original_inputs": {
            "input_ids": [],
            "attention_mask": []
        },
        "masked_inputs": {
            "input_ids": [],
            "attention_mask": []
        },
        # "labels": [],
        "relations": [],
        "original_texts": [],
        "masked_texts": [],
        "subject_entity": [],
        "object_entity": [],
        "subject_type": [],
        "object_type": []
    }

    # Gather items from each instance in the batch
    for instance in batch:
        batch_dict["instance_id"].append(instance["instance_id"])
        
        # Original inputs
        batch_dict["original_inputs"]["input_ids"].append(instance["original_inputs"]["input_ids"])
        batch_dict["original_inputs"]["attention_mask"].append(instance["original_inputs"]["attention_mask"])
        
        # Masked inputs
        batch_dict["masked_inputs"]["input_ids"].append(instance["masked_inputs"]["input_ids"])
        batch_dict["masked_inputs"]["attention_mask"].append(instance["masked_inputs"]["attention_mask"])
        
        # Other fields
        # batch_dict["labels"].append(instance["label"])
        batch_dict["relations"].append(instance["relation"])
        batch_dict["original_texts"].append(instance["original_text"])
        batch_dict["masked_texts"].append(instance["masked_text"])

        # Entity information
        batch_dict["subject_entity"].append(instance["subject_entity"])
        batch_dict["object_entity"].append(instance["object_entity"])
        batch_dict["subject_type"].append(instance["subject_type"])
        batch_dict["object_type"].append(instance["object_type"])
    
    # Convert lists to tensors where appropriate
    batch_dict["instance_id"] = torch.tensor(batch_dict["instance_id"])
    
    # Use pad_sequence instead of stack for variable-length sequences
    batch_dict["original_inputs"]["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        batch_dict["original_inputs"]["input_ids"], batch_first=True, padding_value=0
    )
    batch_dict["original_inputs"]["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        batch_dict["original_inputs"]["attention_mask"], batch_first=True, padding_value=0
    )
    batch_dict["masked_inputs"]["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        batch_dict["masked_inputs"]["input_ids"], batch_first=True, padding_value=0
    )
    batch_dict["masked_inputs"]["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        batch_dict["masked_inputs"]["attention_mask"], batch_first=True, padding_value=0
    )
    
    # batch_dict["labels"] = torch.tensor(batch_dict["labels"])
    
    return batch_dict


def custom_metrics(ground_truths, predictions):
    # Create mapping of relation types to IDs, ensuring 'no_relation' is mapped to 0
    unique_relations = sorted(set(ground_truths) | set(predictions))
    relation_to_id = {}

    relation_to_id = {'no_relation': 0}
    relation_to_id.update({rel: idx + 1 for idx, rel in enumerate([r for r in unique_relations if r != 'no_relation'])})
    
    key_ids = np.array([relation_to_id[k] for k in ground_truths])
    pred_ids = np.array([relation_to_id[p] for p in predictions])
    
    correct_by_relation = ((key_ids == pred_ids) & (pred_ids != 0)).sum()
    guessed_by_relation = (pred_ids != 0).sum()
    gold_by_relation = (key_ids != 0).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)

    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        
    return prec_micro, recall_micro, f1_micro

def compute_metrics(predictions, ground_truths):
    """Compute precision, recall, F1 scores and accuracy"""
    # Convert to lists if they're not already
    y_true = list(ground_truths)
    y_pred = list(predictions)

    labels = sorted(set(y_true) | set(y_pred))
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)

    precision_weight, recall_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0)

    precision_custom, recall_custom, f1_custom = custom_metrics(y_true, y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision_micro': precision_micro, 
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weight,
        'recall_weighted': recall_weight,
        'f1_weighted': f1_weight,  
        'precision_custom': precision_custom,
        'recall_custom': recall_custom,
        'f1_custom': f1_custom,  


    }