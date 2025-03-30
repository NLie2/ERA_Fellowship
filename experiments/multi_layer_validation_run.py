#!/usr/bin/env python3
# Multi-layer validation run script for causal intervention parameters

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

def run_control(model, tokenizer, validation_df, output_dir, validation_name):
    """Run the model without any intervention as a control."""
    print(f"Running control (no intervention) for {validation_name}...")
    
    output_path = output_dir / f"{validation_name}_no_intervention_control.pkl"
    result_path = output_dir / f"{validation_name}_no_intervention_control_rated.pkl"
    
    # Check if already processed
    if result_path.exists():
        print(f"Skipping control for {validation_name} - already processed")
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
            prompt = row["original_prompt_text"]  # Fallback
        
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

def run_multi_layer_intervention(probe_config_row, model, tokenizer, validation_df, layer_index, layer_name, input_size, output_dir, validation_name):
    """Run the model with intervention at a specific layer using probe configuration."""
    probe_type = probe_config_row['probe_type']
    od = probe_config_row['offensive_defensive']
    c_strength = probe_config_row['c_strength']
    lr = probe_config_row['lr'] if not pd.isna(probe_config_row['lr']) else None
    probe_path = probe_config_row['probe_path']
    
    print(f"Running {layer_name} layer intervention with {probe_type} probe, {od}, c_strength={c_strength}, lr={lr}")
    
    # Create unique identifier for this configuration
    config_id = f"{validation_name}_{probe_type}_{od}_c{c_strength}_lr{lr}_layer{layer_index}_{layer_name}"
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
    response_column = f"response_{probe_type}_cstrength={c_strength}_lr={lr}_layer{layer_index}_{layer_name}"
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
            prompt = row["original_prompt_text"]  # Fallback
        
        # Apply chat template
        formatted_prompt = apply_chat_template(tokenizer, prompt)
        
        # Get response using appropriate function with formatted prompt
        if probe_type == "linear":
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                probe=probe,
                layers_to_intervene=[layer_index],
                intervention_strength=c_strength
            )
        else:
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                probe=probe,
                layers_to_intervene=[layer_index],
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

def grade_responses(df, autograder, autogr_tokenizer, response_column, output_path):
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
            row["original_prompt_text"],  # Use original_prompt_text for grading
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

def load_validation_file(validation_path):
    """Load validation file and perform basic checks."""
    if not os.path.exists(validation_path):
        print(f"Validation file not found: {validation_path}")
        return None
    
    try:
        df = pd.read_csv(validation_path)
        print(f"Loaded validation dataset with {len(df)} samples")
        
        # Ensure original_prompt_text column exists
        if 'original_prompt_text' not in df.columns:
            print("Warning: original_prompt_text column not found in validation file")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading validation file: {e}")
        return None

def run_validation_file(model_name, validation_path, validation_name, params_df, model, tokenizer, 
                      autograder, autogr_tokenizer, output_dir, num_layers, input_size):
    """Run validation on a specific validation file."""
    # Load validation file
    validation_df = load_validation_file(validation_path)
    if validation_df is None:
        return []
    
    # Create subdirectory for this validation file
    validation_output_dir = output_dir / validation_name
    os.makedirs(validation_output_dir, exist_ok=True)
    
    # Run control experiment (no intervention)
    control_df = run_control(model, tokenizer, validation_df, validation_output_dir, validation_name)
    
    # Grade control responses
    control_rated_path = validation_output_dir / f"{validation_name}_no_intervention_control_rated.pkl"
    control_rated_df = grade_responses(
        control_df,
        autograder,
        autogr_tokenizer,
        "control_no_intervention_response",
        control_rated_path
    )
    
    # Calculate and store baseline ASR
    baseline_asr = calculate_asr(control_rated_df, "control_no_intervention_response")
    
    # Define layer configurations - early, middle, and late layers
    early_layer = num_layers // 4
    middle_layer = num_layers // 2
    late_layer = (num_layers * 3) // 4
    
    layer_configs = [
        (early_layer, "early"),
        (middle_layer, "middle"),
        (late_layer, "late")
    ]
    
    # Results collection for this validation file
    results = []
    
    # Get intervention configurations
    intervention_rows = params_df[
        (params_df['probe_type'].notna()) & 
        (params_df['probe_type'] != 'control') & 
        (params_df['offensive_defensive'].notna())
    ]
    
    # Run interventions for each layer configuration
    for layer_index, layer_name in layer_configs:
        print(f"Running {layer_name} layer interventions (layer {layer_index}) for {validation_name}")
        
        for _, row in intervention_rows.iterrows():
            probe_type = row['probe_type']
            od = row['offensive_defensive']
            c_strength = row['c_strength']
            lr = row['lr'] if 'lr' in row and not pd.isna(row['lr']) else None
            
            # Run the intervention for this layer
            intervention_df = run_multi_layer_intervention(
                row, 
                model, 
                tokenizer, 
                validation_df, 
                layer_index, 
                layer_name, 
                input_size, 
                validation_output_dir,
                validation_name
            )
            
            # Grade responses
            response_column = f"response_{probe_type}_cstrength={c_strength}_lr={lr}_layer{layer_index}_{layer_name}"
            rated_path = validation_output_dir / f"{validation_name}_{probe_type}_{od}_c{c_strength}_lr{lr}_layer{layer_index}_{layer_name}_rated.pkl"
            
            intervention_rated_df = grade_responses(
                intervention_df,
                autograder,
                autogr_tokenizer,
                response_column,
                rated_path
            )
            
            # Calculate ASR
            asr = calculate_asr(intervention_rated_df, response_column)
            
            # Store results
            result = {
                "model": model_name,
                "validation_file": validation_name,
                "probe_type": probe_type,
                "offensive_defensive": od,
                "c_strength": c_strength,
                "lr": lr,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "asr": asr,
                "baseline_asr": baseline_asr
            }
            results.append(result)
            
            # Save interim results
            interim_results_df = pd.DataFrame(results)
            interim_results_path = validation_output_dir / f"{validation_name}_interim_results.csv"
            interim_results_df.to_csv(interim_results_path, index=False)
    
    # Save final results for this validation file
    validation_results_df = pd.DataFrame(results)
    validation_results_path = validation_output_dir / f"{validation_name}_results.csv"
    validation_results_df.to_csv(validation_results_path, index=False)
    print(f"Results for {validation_name} saved to {validation_results_path}")
    
    return results

def main(model_name, params_csv_path, validation_files):
    """Main function to run the multi-layer validation on multiple validation files."""
    # Load the parameters from the CSV file
    if not os.path.exists(params_csv_path):
        print(f"Parameters file not found: {params_csv_path}")
        return
    
    print(f"Loading parameters from {params_csv_path}")
    params_df = pd.read_csv(params_csv_path)
    
    # Get model details
    model_short_name, num_layers, rating_column, hf_ident, input_size = get_model_details(model_name)
    print(f"Model: {model_short_name}, Layers: {num_layers}, Input size: {input_size}")
    
    # Load the model and tokenizer
    model, tokenizer = get_model(hf_ident)
    
    # Load the autograder
    autogr_name = "cais/HarmBench-Llama-2-13b-cls"
    autograder = AutoModelForCausalLM.from_pretrained(autogr_name)
    autogr_tokenizer = AutoTokenizer.from_pretrained(autogr_name, use_fast=False, truncation_side="left", padding_side="left")
    
    # Setup main output directory
    output_dir = PROJECT_ROOT / f"datasets/validation_results/{model_short_name}/multi_layer"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each validation file
    all_results = []
    
    for val_file in validation_files:
        # Extract the file name without extension to use as validation name
        validation_name = Path(val_file).stem
        validation_path = PROJECT_ROOT / val_file
        
        print(f"\n{'='*80}\nProcessing validation file: {validation_name}\n{'='*80}")
        
        # Run validation on this file
        results = run_validation_file(
            model_name,
            validation_path,
            validation_name,
            params_df,
            model,
            tokenizer,
            autograder,
            autogr_tokenizer,
            output_dir,
            num_layers,
            input_size
        )
        
        all_results.extend(results)
    
    # Save combined results
    combined_results_df = pd.DataFrame(all_results)
    combined_results_path = output_dir / "combined_results.csv"
    combined_results_df.to_csv(combined_results_path, index=False)
    print(f"\nCombined results saved to {combined_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-layer validation with causal intervention parameters on multiple validation files.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to validate")
    parser.add_argument("--params_csv", type=str, required=False, 
                        default="datasets/public_2/gemma-7b-it/intervention/causal_intervention_parameters_with_capability_and_coherence_multilayer.csv",
                        help="Path to the parameters CSV file")
    parser.add_argument("--validation_files", type=str, nargs='+', required=True,
                        help="Paths to validation CSV files (relative to PROJECT_ROOT)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    params_csv_path = PROJECT_ROOT / args.params_csv
    
    main(args.model_name, params_csv_path, args.validation_files) 