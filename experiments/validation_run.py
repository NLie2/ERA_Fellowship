#!/usr/bin/env python3
# Validation run script for causal intervention parameters

import pandas as pd
import numpy as np
import torch
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.causal_interventions import *
from analysis.utils import *
from analysis.models import get_model_details
from gen_data.harmBench_autograder import grade_with_HF
from lm_eval.models.huggingface import HFLM

# Import paths module
from experiments.paths import PROJECT_ROOT, DATASETS_DIR, set_project_root_as_cwd

# Set project root as current working directory
set_project_root_as_cwd()

# Configure the probe types
PROBE_CONFIGS = {
    "linear": {
        "loader": lambda path: pickle.load(open(path, 'rb')),
        "response_fn": get_response_linear,
        "probe_class": None  # Linear probe doesn't need initialization
    },
    "mlp": {
        "loader": lambda path, input_size: SimpleMLP(input_size=input_size).load_probe(path),
        "response_fn": get_response_MLP,
        "probe_class": SimpleMLP
    },
    "transformer": {
        "loader": lambda path, input_size: TransformerProbe(input_size=input_size).load_probe(path),
        "response_fn": get_response_transformer,
        "probe_class": TransformerProbe
    }
}

def get_pad_token(model_id, tokenizer):
    """Get the appropriate pad token for the model."""
    if tokenizer.pad_token is not None:
        return tokenizer.pad_token
    
    model_to_pad_token = {
        "meta-llama/Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>",
        "google/gemma-7b-it": "<pad>",
        "mistralai/Ministral-8B-Instruct-2410": "<pad>",
        "meta-llama/Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>"
    }
    
    pad_token = model_to_pad_token.get(model_id, "<pad>")  # Default to <pad> if not found
    print(f"Using pad token for {model_id}: {pad_token}")
    return pad_token

def get_model(model_name: str):
    """Load the model and tokenizer with the correct configuration."""
    # Load the model with float16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set the pad token using the same logic as control experiments
    tokenizer.pad_token = get_pad_token(model_name, tokenizer)
    print(f"Pad token set to: {tokenizer.pad_token}")

    # Print the model's dtype for debugging
    model_dtype = next(model.parameters()).dtype
    print(f"Model loaded with dtype: {model_dtype}")
    
    return model, tokenizer

def apply_chat_template(tokenizer, prompt):
    """Apply the chat template to the prompt."""
    # Format the prompt according to the chat template
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    return formatted

def run_control(model, tokenizer, validation_df, output_dir):
    """Run the model without any intervention as a control."""
    print("Running control (no intervention)...")
    
    output_path = output_dir / "no_intervention_control.pkl"
    result_path = output_dir / "no_intervention_control_rated.pkl"
    
    # Check if already processed
    if result_path.exists():
        print(f"Skipping control - already processed")
        return pd.read_pickle(result_path)
    
    # Copy the validation dataframe to avoid modifying the original
    df = validation_df.copy()
    df["control_no_intervention_response"] = None
    
    # Process each prompt
    model.cuda()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Get the prompt - check for different column names
        if "jailbreak_prompt" in row:
            prompt = row["jailbreak_prompt"]
        elif "jailbreak_prompt_text" in row:
            prompt = row["jailbreak_prompt_text"]
        else:
            prompt = row["original_prompt"]  # Fallback
        
        # Apply chat template
        formatted_prompt = apply_chat_template(tokenizer, prompt)
        
        # Tokenize the formatted prompt
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**input_ids, max_new_tokens=200)
            response = tokenizer.decode(outputs[0])

        # Extract response text by removing the prompt
        current_prompt = formatted_prompt
        if "user\n" in response:
            parts = response.split("user\n")
            if len(parts) > 1:
                response = parts[-1].split("\n", 1)
                response = response[1].strip() if len(response) > 1 else response[0].strip()
        else:
            response = response[len(current_prompt):].strip() if response.startswith(current_prompt) else response
                
        df.at[index, "control_no_intervention_response"] = response
        
        # Save periodically
        if index % 20 == 0:
            os.makedirs(output_dir, exist_ok=True)
            df.to_pickle(output_path)
    
    # Save final results
    os.makedirs(output_dir, exist_ok=True)
    df.to_pickle(output_path)
    print(f"Control data saved to {output_path}")
    
    model.cpu()
    return df

def run_intervention(probe_config_row, model, tokenizer, validation_df, middle_layer, input_size, output_dir):
    """Run the model with intervention using probe configuration."""
    probe_type = probe_config_row['probe_type']
    od = probe_config_row['offensive_defensive']
    c_strength = probe_config_row['c_strength']
    lr = probe_config_row['lr'] if not pd.isna(probe_config_row['lr']) else None
    probe_path = probe_config_row['probe_path']
    
    print(f"Running intervention with {probe_type} probe, {od}, c_strength={c_strength}, lr={lr}")
    
    # Create unique identifier for this configuration
    config_id = f"{probe_type}_{od}_c{c_strength}_lr{lr}"
    output_path = output_dir / f"{config_id}.pkl"
    
    # Check if already processed
    if output_path.exists():
        print(f"Skipping {config_id} - already processed")
        return pd.read_pickle(output_path)
    
    # Setup model
    model.cuda()
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    # Load the probe
    config = PROBE_CONFIGS[probe_type]
    if probe_type == "linear":
        probe = config["loader"](probe_path)
        if hasattr(probe, 'coef_') and isinstance(probe.coef_, np.ndarray):
            probe.coef_ = probe.coef_.astype(np.float16 if model_dtype == torch.float16 else np.float32)
    else:
        probe = config["probe_class"](input_size=input_size)
        probe.load_probe(probe_path)
        probe.to(device=device, dtype=model_dtype)
    
    # Copy the validation dataframe to avoid modifying the original
    df = validation_df.copy()
    response_column = f"response_{probe_type}_cstrength={c_strength}_lr={lr}"
    df[response_column] = None
    
    # Process each prompt
    offensive = True if od == "offensive" else False
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Get the prompt - check for different column names
        if "jailbreak_prompt" in row:
            prompt = row["jailbreak_prompt"]
        elif "jailbreak_prompt_text" in row:
            prompt = row["jailbreak_prompt_text"]
        else:
            prompt = row["original_prompt"]  # Fallback
        
        # Apply chat template
        formatted_prompt = apply_chat_template(tokenizer, prompt)
        
        # Get response using appropriate function with formatted prompt
        if probe_type == "linear":
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                probe=probe,
                layers_to_intervene=[middle_layer],
                intervention_strength=c_strength
            )
        else:
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                probe=probe,
                layers_to_intervene=[middle_layer],
                c=c_strength,
                learning_rate=lr,
                offensive=offensive
            )
        
        df.at[index, response_column] = response
        
        # Save periodically
        if index % 20 == 0:
            os.makedirs(output_dir, exist_ok=True)
            df.to_pickle(output_path)
    
    # Save final results
    os.makedirs(output_dir, exist_ok=True)
    df.to_pickle(output_path)
    print(f"Intervention data saved to {output_path}")
    
    model.cpu()
    return df

def grade_responses(df, autograder, autogr_tokenizer, request_column, response_column, output_path):
    """Grade the responses using the autograder."""
    print(f"Grading responses for column: {response_column}")
    
    # Check if the result already exists
    if output_path.exists():
        print(f"Skipping grading - result already exists at {output_path}")
        return pd.read_pickle(output_path)
    
    # Create a rating column
    rating_column = f"rating_{response_column}"
    
    # Move the autograder to GPU
    autograder.cuda()
    
    # Apply the grading function row by row
    df[rating_column] = df.apply(
        lambda row: grade_with_HF(
            row[request_column], 
            [row[response_column]], 
            autograder, 
            autogr_tokenizer, 
            verbose=True
        )[0] if pd.notna(row[response_column]) else "No Response", 
        axis=1
    )
    
    # Save the results
    df.to_pickle(output_path)
    print(f"Graded results saved to {output_path}")
    
    # Move the autograder back to CPU
    autograder.cpu()
    
    return df

def calculate_asr(df, response_column):
    """Calculate the Attack Success Rate from the graded responses."""
    rating_column = f"rating_{response_column}"
    
    if rating_column not in df.columns:
        print(f"Rating column {rating_column} not found in dataframe")
        return 0
    
    # Count 'Yes' responses as successful attacks
    yes_count = df[rating_column].value_counts().get("Yes", 0)
    total_valid = len(df[df[rating_column].notna()])
    
    if total_valid == 0:
        return 0
    
    # Calculate percentage
    asr = (yes_count / total_valid) * 100
    print(f"ASR for {response_column}: {asr:.2f}%")
    
    return asr

def main(model_name, params_csv_path):
    """Main function to run the validation."""
    # Load the parameters from the CSV file
    if not os.path.exists(params_csv_path):
        print(f"Parameters file not found: {params_csv_path}")
        return
    
    print(f"Loading parameters from {params_csv_path}")
    params_df = pd.read_csv(params_csv_path)
    
    # Get model details
    model_short_name, num_layers, rating_column, hf_ident, input_size = get_model_details(model_name)
    print(f"Model: {model_short_name}, Layers: {num_layers}, Input size: {input_size}")
    
    # Get middle layer (for intervention)
    middle_layer = num_layers // 2
    print(f"Using middle layer: {middle_layer}")
    
    # Load the model and tokenizer
    model, tokenizer = get_model(hf_ident)
    
    # Load the autograder
    autogr_name = "cais/HarmBench-Llama-2-13b-cls"
    autograder = AutoModelForCausalLM.from_pretrained(autogr_name)
    autogr_tokenizer = AutoTokenizer.from_pretrained(autogr_name, use_fast=False, truncation_side="left", padding_side="left")
    
    # Load the validation dataset
    validation_path = PROJECT_ROOT / "datasets/validation/jailbreak_bench.csv"
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded validation dataset with {len(validation_df)} samples")
    
    # Setup output directory
    output_dir = PROJECT_ROOT / f"datasets/validation_results/{model_short_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run control experiment (no intervention)
    control_df = run_control(model, tokenizer, validation_df, output_dir)
    
    # Grade control responses
    control_rated_path = output_dir / "no_intervention_control_rated.pkl"
    control_rated_df = grade_responses(
        control_df,
        autograder,
        autogr_tokenizer,
        "original_prompt" if "original_prompt" in validation_df.columns else "jailbreak_prompt",
        "control_no_intervention_response",
        control_rated_path
    )
    
    # Calculate and store baseline ASR
    baseline_asr = calculate_asr(control_rated_df, "control_no_intervention_response")
    
    # Results collection
    results = []
    
    # Run each intervention from the parameters CSV
    # Filter to only rows with probe type and offensive_defensive columns (non-control rows)
    intervention_rows = params_df[
        (params_df['probe_type'].notna()) & 
        (params_df['probe_type'] != 'control') & 
        (params_df['offensive_defensive'].notna())
    ]
    
    for _, row in intervention_rows.iterrows():
        probe_type = row['probe_type']
        od = row['offensive_defensive']
        c_strength = row['c_strength']
        lr = row['lr'] if 'lr' in row and not pd.isna(row['lr']) else None
        
        # Create config ID
        config_id = f"{probe_type}_{od}_c{c_strength}_lr{lr}"
        
        # Output paths
        intervention_path = output_dir / f"{config_id}.pkl"
        intervention_rated_path = output_dir / f"{config_id}_rated.pkl"
        
        # Run intervention
        intervention_df = run_intervention(
            row, 
            model, 
            tokenizer, 
            validation_df, 
            middle_layer, 
            input_size, 
            output_dir
        )
        
        # Grade responses
        response_column = f"response_{probe_type}_cstrength={c_strength}_lr={lr}"
        intervention_rated_df = grade_responses(
            intervention_df,
            autograder,
            autogr_tokenizer,
            "original_prompt" if "original_prompt" in validation_df.columns else "jailbreak_prompt",
            response_column,
            intervention_rated_path
        )
        
        # Calculate ASR
        asr = calculate_asr(intervention_rated_df, response_column)
        
        # Store results
        result = {
            "model": model_short_name,
            "probe_type": probe_type,
            "offensive_defensive": od,
            "c_strength": c_strength,
            "lr": lr,
            "asr": asr,
            "baseline_asr": baseline_asr
        }
        results.append(result)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = output_dir / "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Validation results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation with causal intervention parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to validate")
    parser.add_argument("--params_csv", type=str, required=False, 
                        default="datasets/public_2/gemma-7b-it/intervention/causal_intervention_parameters_with_capability_and_coherence_multilayer.csv",
                        help="Path to the parameters CSV file")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    params_csv_path = PROJECT_ROOT / args.params_csv
    
    main(args.model_name, params_csv_path) 