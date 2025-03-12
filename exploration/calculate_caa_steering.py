import torch
import pyvene as pv
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-2-2b-it"
layer = 10


def load_prompts(positive_paths: list[str], negative_paths: list[str]):
    positive_prompts = []
    for path in positive_paths:
        with open(f"exploration/data/{path}.txt", "r") as file:
            positive_prompts.extend(file.read().splitlines())
            
    negative_prompts = []
    for path in negative_paths:
        with open(f"exploration/data/{path}.txt", "r") as file:
            negative_prompts.extend(file.read().splitlines())
    
    return tokenizer(positive_prompts, return_tensors="pt", padding=True), tokenizer(negative_prompts, return_tensors="pt", padding=True)


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))
    positive_prompts, negative_prompts = load_prompts(
        positive_paths = ["animals_True_steering", "countries_True_steering"],
        negative_paths = ["animals_False_steering", "countries_False_steering"])
    pv_gemma = pv.IntervenableModel(
        {
            "layer": 10,
            "component": "mlp_output",
            "intervention_type": pv.CollectIntervention
        },
        model=model 
    )

    positive_activations = pv_gemma(
        base=positive_prompts,
        unit_locations={
            "base": [int(positive_prompts["input_ids"].shape[-1])-1]
        }
    )

    positive_representation = torch.mean(torch.stack(positive_activations[0][-1], dim = 0), dim=0)

    negative_activations = pv_gemma(
        base=positive_prompts,
        unit_locations={
            "base": [int(negative_prompts["input_ids"].shape[-1])-1]
        }
    )

    negative_representation = torch.mean(torch.stack(negative_activations[0][-1], dim = 0), dim=0)
    steering_vector = positive_representation - negative_representation
    torch.save(steering_vector, f"exploration/tensors/{model_name}_steering_vector.pt")