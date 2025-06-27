import torch
from neural_networks.modeling.train import main as train

train(False, True, "cnn", 5, 5e-4, "runs/final_best", 64, 128)
train(False, True, "rnn", 5, 5e-4, "runs/final_best", 64, 32)

state_dict = torch.load("Common-voice-asr/neural_networks/runs/final_best/best_cnn.pth")
torch.save(state_dict, "Common-voice-asr/models/best_cnn.pth")
state_dict = torch.load("Common-voice-asr/neural_networks/runs/final_best/best_rnn.pth")
torch.save(state_dict, "Common-voice-asr/models/best_rnn.pth")
