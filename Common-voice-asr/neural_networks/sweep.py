# Create sweep: wandb sweep Common-voice-asr/neural_networks/configs/week5_sweep.yaml
# Run sweep agent with: wandb agent username/project_name/project_id
# python Common-voice-asr/neural_networks/sweep.py
# python neural_networks/sweep.py --model [x]
import wandb
import os
import yaml
import argparse
from neural_networks.modeling.train import main as train
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))
logdir_path_a = os.path.join(BASE_DIR, "neural_networks/runs/week8_sweep_a")
logdir_path_b = os.path.join(BASE_DIR, "neural_networks/runs/week8_sweep_b")

config_path_a = os.path.join(BASE_DIR, "neural_networks/configs/week8_sweep_a.yaml")
with open(config_path_a) as f:
    sweep_config_path_a = yaml.safe_load(f)
    
config_path_b = os.path.join(BASE_DIR, "neural_networks/configs/week8_sweep_b.yaml")
with open(config_path_b) as f:
    sweep_config_path_b = yaml.safe_load(f)
    
config_path_spects = os.path.join(BASE_DIR, "neural_networks/configs/week8_sweep_mels.yaml")
with open(config_path_spects) as f:
    sweep_config_path_spects = yaml.safe_load(f)

    
def parse_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['a', 'b'], required=True, help='Specify which model to sweep. a: RNN & CNN or b: Hybrid Transformer')
    parser.add_argument('--sample_spects', action='store_true', help='Train & validate with a mel-spectogram sweep')
    return parser.parse_args()

def sweep_train_modelA():
    try:
        with wandb.init():
            config = wandb.config
            run_id = wandb.run.id
            logdir = os.path.join(logdir_path_a,
                                  f"lr{config.learning_rate}_bs{config.batch_size}_hd{config.hidden_dimension}_model{config.model_type}")
            wandb.config.update({"logdir": logdir})
            Path(logdir).mkdir(parents=True, exist_ok=True)
            check_data = False
            train(check_data, config.full_mini, config.corpus, config.greedy, config.model_type, config.epochs, config.learning_rate, 
                  logdir, config.batch_size, config.hidden_dimension, lm_weight=config.lm_weight, word_score=config.word_score,
                  sample_size=config.sample_size, sample_spect_folder=config.sample_spect_folder)
    except Exception as e:
        print(f"[ERROR] Run failed with error: {e}")
        wandb.finish(exit_code=1)

        
def sweep_train_modelB():
    try:
        with wandb.init():
            config = wandb.config
            run_id = wandb.run.id
            logdir = os.path.join(logdir_path_b,
                                  f"lr{config.learning_rate}_bs{config.batch_size}_hd{config.hidden_dimension}_model{config.model_type}")
            wandb.config.update({"logdir": logdir})
            Path(logdir).mkdir(parents=True, exist_ok=True)
            check_data = False
            train(check_data, config.full_mini, config.corpus, config.greedy, config.model_type, config.epochs, config.learning_rate, 
                  logdir, config.batch_size, config.hidden_dimension, lm_weight=config.lm_weight, word_score=config.word_score,
                  sample_size=config.sample_size)
    except Exception as e:
        print(f"[ERROR] Run failed with error: {e}")
        wandb.finish(exit_code=1)


def main(model: str = 'a', sample_spects: bool = False):
    if sample_spects:
        sweep_id = wandb.sweep(sweep_config_path_spects, project="week8_sweep_mels")
        function = sweep_train_modelA
        wandb.agent(sweep_id, function, count=5)
    else:
        if model == 'a':
            sweep_id = wandb.sweep(sweep_config_path_a, project="week8_sweep_a")
            wandb.agent(sweep_id, sweep_train_modelA, count=24)
        elif model == 'b':
            sweep_id = wandb.sweep(sweep_config_path_b, project="week8_sweep_b")
            wandb.agent(sweep_id, sweep_train_modelB, count=100)
        else:
            print("Model can only be a or b")
            return


if __name__ == "__main__":
    args = parse_command_args()
    main(args.model, args.sample_spects)
