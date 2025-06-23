import os
import yaml
import wandb
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from mockito import when, verify, unstub

from neural_networks.datasets import CTC_MiniCVDataset
from neural_networks.modeling.train import main as train 

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

def test_load_sweep():
    config_path = os.path.join(BASE_DIR, "neural_networks/configs/week5_sweep.yaml")
    assert os.path.getsize(config_path) > 0, "Config did not load correctly, file is empty"

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)
    assert "method" in sweep_config
    assert "metric" in sweep_config
    expected_keys = {"learning_rate", "batch_size", "hidden_dimension", "epochs", "model_type", "full_mini"}
    actual_keys = set(sweep_config["parameters"].keys())
    assert "parameters" in sweep_config
    assert actual_keys == expected_keys, f"Expected parameter keys {expected_keys}, instead they are: {actual_keys}"

def test_hyperparameter_presence():
    config_path = os.path.join(BASE_DIR, "neural_networks/configs/week5_sweep.yaml")
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)
    parameters = sweep_config["parameters"]
    for param_name, param_vals in parameters.items():
        assert isinstance(param_vals, dict), f"{param_name} should be a dictionary with the format values : [val1, val2]"
        assert any(key in param_vals and param_vals[key] not in ([], None) for key in ["values"]), f"{param_name} must specify values"
        
dummy_config = {"method" : "random", "metric" : {"name" : "val/wer", "goal" : "minimize"}, "parameters" : 
                {"learning_rate" : {"values" : [0.0001]}, "batch_size" : {"values" : [4]}, "hidden_dimension" : {"values" : [32]}}}

def sweep_train_dummy():
    with wandb.init():
            config = wandb.config
            assert config.learning_rate == 0.0001, "Incorrect LR"
            assert config.batch_size == 4, "Incorrect batch size"
            assert config.hidden_dimension == 32, "Incorrect hidden dimension val"
            train(False, True, "cnn", 1, config.learning_rate, "runs/test_sweep", config.batch_size, config.hidden_dimension, True)

def test_dummy_sweep():
    try:
        when(wandb).sweep(dummy_config, project="sweep_test").thenReturn("mock_sweep_id")
        when(wandb).agent("mock_sweep_id", function=sweep_train_dummy, count=1).thenReturn(None)

        sweep_id = wandb.sweep(dummy_config, project="sweep_test")
        wandb.agent(sweep_id, function=sweep_train_dummy, count=1)
        verify(wandb).sweep(dummy_config, project="sweep_test")
        verify(wandb).agent("mock_sweep_id", function=sweep_train_dummy, count=1)
    finally:
        unstub()

if __name__ == "__main__":
    test_load_sweep()
    test_hyperparameter_presence()
    test_dummy_sweep()
    print("All sweep tests passed.")
