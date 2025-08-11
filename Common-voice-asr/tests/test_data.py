from neural_networks.modeling.train import main
from argparse import Namespace

args = Namespace(model_type="rnn", logdir="runs/test_train_runs", check_data=True, full_mini=False, corpus=False, greedy=False,
                 epochs=1, lr=1e-3, batch_size=2, hidden_dim=128, test_sweep=False, lm_weight=3.23, word_score=-0.26,
                 sample_size=100, d_model=256, nhead=2, dim_feedforward=512, nlayers=2, lstm_hidden=128, lstm_layers=1,
                 dropout=0.3, sample_spect_folder=None, debug_sample=False, conv_layer=False, beam_width=4, best=True, demo=False)


# sample pytest for test pass
def test_code_is_tested():
    assert 1 + 1 == 2


def test_train_stub_runs():
    main(args)
