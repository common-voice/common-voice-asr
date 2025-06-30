# Create sweep: wandb sweep Common-voice-asr/neural_networks/configs/week5_sweep.yaml
# Run sweep agent with: wandb agent username/project_name/project_id
# python Common-voice-asr/neural_networks/sweep.py
# python neural_networks/sweep.py
import wandb
import os
import yaml
from neural_networks.modeling.train import main as train
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

config_path = os.path.join(BASE_DIR, "neural_networks/configs/week6_sweep.yaml")
with open(config_path) as f:
    sweep_config = yaml.safe_load(f)

def sweep_train():
    try:
        with wandb.init():
            config = wandb.config
            run_id = wandb.run.id
            logdir = os.path.join(BASE_DIR, f"neural_networks/runs/week6_beam/sweep_rnn_100/lr{config.learning_rate}_bs{config.batch_size}_hd{config.hidden_dimension}_run{run_id}")
            wandb.config.update({"logdir": logdir})
            Path(logdir).mkdir(parents=True, exist_ok=True)
            train(check_data=False, full_mini=config.full_mini, model_type=config.model_type, epochs= config.epochs, lr = config.learning_rate, 
                logdir = logdir, batch_size = config.batch_size, hidden_dim = config.hidden_dimension, lm_weight=config.lm_weight, word_score=config.word_score)
    except Exception as e:
        print(f"[ERROR] Run failed with error: {e}")
        wandb.finish(exit_code=1)
    
def main():
    sweep_id = wandb.sweep(sweep_config, project="week6_beam")
    wandb.agent(sweep_id, sweep_train, count=100)        

if __name__ == "__main__":
    main()