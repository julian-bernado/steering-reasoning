import torch
import numpy as np
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from pyvene import IntervenableModel, IntervenableConfig, RepresentationConfig
from pyvene.models.interventions import Intervention

class SteeringVectorIntervention(Intervention):
    def __init__(self, steering_vector, alpha = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.alpha = alpha
        
    def forward(self, base, source=None, subspaces=None):
        return base + self.alpha * self.steering_vector

def create_intervenable_model(model, steering_vector, layer=10, alpha=1.0):
    intervention = SteeringVectorIntervention(steering_vector=steering_vector, alpha=alpha)
    print("Model type is ")
    print(type(model))
    
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer,
                "mlp_output",
                "pos",
                intervention=intervention
            )
        ]
    )
    
    return IntervenableModel(config, model)

def generate_text(model, tokenizer, prompt, steering_vector=None, max_length=100, top_p=0.9, alpha=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if steering_vector is not None:
        intervenable_model = create_intervenable_model(model, steering_vector, alpha=alpha)
    else:
        intervenable_model = model
    
    generated = input_ids
    max_seq_length = min(max_length, model.config.max_position_embeddings)
    
    for _ in range(max_length - len(input_ids[0])):
        if generated.shape[1] > max_seq_length:
            generated = generated[:, -max_seq_length:]
        
        with torch.no_grad():
            if steering_vector is not None:
                outputs = intervenable_model({"input_ids": generated}, sources=None, unit_locations={"base": generated.shape[1]-1})[1]
            else:
                outputs = model(generated)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            next_token_logits = logits[:, -1, :].squeeze()
            next_token_logits = next_token_logits
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    model_name = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    steering_vector = torch.load("exploration/tensors/google/gemma-2-2b-it_steering_vector.pt")
    print(f"Loaded steering vector with shape: {steering_vector.shape}")
    alpha = -10.0
    
    print("Enter a prompt, and see the difference between the original and steered outputs.")
    print("Type 'exit' to quit.")
    
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'exit':
            break
        if prompt.lower().startswith("alpha="):
            alpha = float(prompt.split('=')[1])
            print(f"Setting alpha to {alpha}")
        
        print("\nGenerating unsteered output...")
        unsteered_output = generate_text(model, tokenizer, prompt)
        print("\nUnsteered output:")
        print(unsteered_output)
        
        print(f"\nGenerating steered output (alpha={alpha})...")
        steered_output = generate_text(model, tokenizer, prompt, steering_vector, alpha=alpha)
        print("\nSteered output:")
        print(steered_output)

if __name__ == "__main__":
    main()