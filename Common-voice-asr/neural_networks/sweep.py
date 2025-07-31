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
logdir_path_a = os.path.join(BASE_DIR, "neural_networks/runs/sweep_a")
logdir_path_b = os.path.join(BASE_DIR, "neural_networks/runs/sweep_b")

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
    parser.add_argument('--sample_spects', action='store_true', default=False, help='Train & validate with a mel-spectogram sweep')
    parser.add_argument('--config', type=str, required=True, help="Specify config file to run with")
    parser.add_argument('--project_name', type=str, required=True, help="Specify project name")
    parser.add_argument('--count', type=int, default=25, help="Number of runs within a single sweep")
    return parser.parse_args()


def sweep_train_modelA():
    try:
        with wandb.init():
            config = dict(wandb.config)
            run_id = wandb.run.id
            logdir = os.path.join(logdir_path_a,
                                  f"lr{config['learning_rate']}_bs{config['batch_size']}_hd{config['hidden_dimension']}_model{config['model_type']}")
            config['logdir'] = logdir
            wandb.config.update({"logdir": logdir})
            Path(logdir).mkdir(parents=True, exist_ok=True)
            
            args = load_args(config, str(BASE_DIR / "neural_networks/configs/train_defaults.yaml")) 
            train(args)
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
            config['logdir'] = logdir
            wandb.config.update({"logdir": logdir})
            Path(logdir).mkdir(parents=True, exist_ok=True)
            check_data = False
            args = load_args(config, str(BASE_DIR / "neural_networks/configs/train_defaults.yaml")) 
            train(args)
    except Exception as e:
        print(f"[ERROR] Run failed with error: {e}")
        wandb.finish(exit_code=1)


def load_args(sweep_config, defaults_path) -> argparse.Namespace:
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
    merged = {**defaults, **sweep_config}
    return argparse.Namespace(**merged)


def load_path(config_path):
    config_path = BASE_DIR / "neural_networks/configs" / config_path
    if not config_path.exists():
        print(f"[ERROR] Config path for your sweep, {config_path}, does not exist")
        exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(args):

    if args.sample_spects:
        config_dict = config_path_spects
    else:
        config_dict = load_path(args.config)

    if args.model == 'a':
        sweep_model = sweep_train_modelA
    elif args.model == 'b':
        sweep_model = sweep_train_modelB
    else:
        print("[ERROR] Model can only be a or b")
        return

    sweep_id = wandb.sweep(config_dict, project=args.project_name)
    wandb.agent(sweep_id, sweep_model, count=args.count)


if __name__ == "__main__":
    args = parse_command_args()
    main(args)
