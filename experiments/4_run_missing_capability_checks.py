import pandas as pd
from pathlib import Path
import argparse

# Assuming the main function is imported from capability_check.py
from analysis.Probes import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
import pickle

from gen_data.harmBench_autograder import grade_with_HF
from analysis.causal_interventions import *
from analysis.models import *

from experiments.paths import PROJECT_ROOT


def check_all_parameters(model):
    return all(p.device.type == 'cuda' for p in model.parameters())


def check_hflm_model_devices(hflm_model):
    print("HF Model device check:", check_all_parameters(hflm_model.model))
    for name, param in hflm_model.model.named_parameters():
        if param.device.type != 'cuda':
            return False
    return True


def run_capability_check(model_name, probe_path=None, c=None, offensive=None, learning_rate=None):
    device = torch.device("cuda")

    model_name, num_layers, rating_column, hf_ident, input_size = get_model_details(model_name)

    model = AutoModelForCausalLM.from_pretrained(hf_ident)
    tokenizer = AutoTokenizer.from_pretrained(hf_ident)

    middle_layer = num_layers // 2

    model.to(device)

    hflm_model = HFLM(model)
    hflm_model.model.to(device)

    assert check_all_parameters(hflm_model.model)

    if probe_path:
        print("OFFENSIVE", offensive, type(offensive))

        offens_defens = "offensive" if offensive else "defensive"

        probe_path = PROJECT_ROOT / probe_path
        if "linear" in str(probe_path):
            probe_type = "linear"

            with open(probe_path, 'rb') as f:
                probe = pickle.load(f)

            offensive = True if c > 0 else False

            insert_linear_hook(model=model, tokenizer=tokenizer, probe=probe, layers_to_intervene=[middle_layer], intervention_strength=c)

        elif "mlp" in str(probe_path):
            probe_type = "mlp"
            probe = SimpleMLP(input_size=input_size)
            probe.load_probe(probe_path)
            probe.to(device)

            assert check_all_parameters(probe)

            insert_mlp_hook(model=model, mlp=probe, layers_to_intervene=[middle_layer], c=c, learning_rate=learning_rate, offensive=offensive)

        elif "transformer" in str(probe_path):
            probe_type = "transformer"
            probe = TransformerProbe(input_size=input_size)

            probe.load_probe(probe_path)
            probe.to(device)

            assert check_all_parameters(probe)

            insert_transformer_hook(model=model, probe=probe, layers_to_intervene=[middle_layer], c=c, learning_rate=learning_rate, offensive=offensive)
            assert check_hflm_model_devices(hflm_model)

    eval_output = evaluator.simple_evaluate(model=hflm_model, tasks=['mmlu'], device='cuda', verbosity='ERROR')

    print(eval_output['results']['mmlu'])

    return eval_output['results']['mmlu']


def run_missing_capability_checks(model_name, base_path, single):
    incomplete_file = Path(base_path) / model_name / "intervention" / f"causal_intervention_parameters_with_capability_and_coherence_{single}.csv"
    if not incomplete_file.exists():
        print(f"Final results file not found: {incomplete_file}")
        return

    df = pd.read_csv(incomplete_file)

    missing_capability_df = df[df['capability'].isna()]

    for index, row in missing_capability_df.iterrows():
        probe_type = row['probe_type']
        offensive_defensive = row['offensive_defensive']
        joint_probe_type = row['joint_probe_type']
        c_strength = row['c_strength']
        lr = row['lr']


        print(f"Probe type: {probe_type}, Offensive/Defensive: {offensive_defensive}, C strength: {c_strength}")

        params_file = Path(base_path) / model_name / "intervention" / f"causal_intervention_parameters_{single}.csv"
        capability_file = Path(base_path) / model_name / "intervention" / f"causal_intervention_parameters_with_capability_{single}.csv"

        if not params_file.exists():
            print(f"Parameters file not found: {params_file}")
            continue

        params_df = pd.read_csv(params_file)
        row = params_df[(params_df["probe_type"] == probe_type) &
                        (params_df["offensive_defensive"] == offensive_defensive) &
                        (params_df["c_strength"] == c_strength)]


        capability_df = pd.read_csv(capability_file)
        cap_row = capability_df[(capability_df["probe_type"] == probe_type) &
                                (capability_df["offensive_defensive"] == offensive_defensive) &
                                (capability_df["c_strength"] == c_strength) &
                                (capability_df["lr"] == lr)]

        if not row.empty:
            probe_path = row.iloc[0]['probe_path']
            mmlu_score = run_capability_check(model_name=model_name, probe_path=probe_path, c=c_strength, offensive=(offensive_defensive == 'offensive'), learning_rate=lr)
            print(f"MMLU score: {mmlu_score}")

            df.at[index, 'capability'] = mmlu_score.get('acc,none', df.at[index, 'capability'])
            df.at[index, 'capability_stderr'] = mmlu_score.get("acc_stderr,none", df.at[index, 'capability_stderr'])

            # also update capability file 
            if not cap_row.empty:
                # Get the index in the original capability DataFrame
                cap_index = cap_row.index[0]
                capability_df.at[cap_index, 'capability'] = mmlu_score.get('acc,none', capability_df.at[cap_index, 'capability'])
                capability_df.at[cap_index, 'capability_stderr'] = mmlu_score.get("acc_stderr,none", capability_df.at[cap_index, 'capability_stderr'])
                
                # Save the updated capability DataFrame
                capability_df.to_csv(capability_file, index=False)
                print(f"Updated capability file: {capability_file}")

            # print the row
            print("updated row: ", df.iloc[index])
        else:
            print("No matching parameters found for the given conditions.")

        df.to_csv(incomplete_file, index=False)
        print("updated dataframe: ", incomplete_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run missing capability checks.')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to use for the capability check.')
    parser.add_argument('--base_path', type=str, required=True, help='The base path where the causal_intervention_parameters file is located.')
    parser.add_argument('--single', type=str, required=True, help='The single layer or multilayer.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_missing_capability_checks(args.model_name, args.base_path, args.single)


if __name__ == '__main__':
    main()

