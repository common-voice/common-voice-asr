# common-voice-asr
Constructing different types of neural networks trained on Common Voice speech datasets to compare the performance of approaches on speech to text performance metrics. 

# Quick-Start
1. Clone the repositiory
    git clone https://github.com/common-voice/common-voice-asr.git
    cd common-voice-asr
    cd Common-voice-asr
2. Create the environment
    make create_environment
    conda activate $(PROJECT_NAME)
3. Install dependencies
    make requirements
4. Train the model
    make train
5. Test the model
    make test

# Week 3 - Quick-Start: Mini-dataset & training
Training:
- Must cd Common-voice-asr to run
- To run train.py, requires model type & number of epochs
    - python -m neural_networks.modeling.train --model_type [type] --epochs # --logdir runs/week3_[type]
    - Model_type must be "cnn" or "rnn" 
- To check data, add flag --check-data before other flags
    - python -m neural_networks.modeling.train --check-data --model_type cnn --epochs 1 --logdir runs/week3_cnn
- Do not be afraid about the Error - No such option: --model_type, it still works fine

Jupyter Notebook: 
- Under notebooks folder: 04_first_cnn_rnn.ipynb - Run All offers demos on displaying a spectogram, running a spectogram, launching training, and plotting logged loss curves.

# Week 4: CTC Training
Within common-voice-asr, run:
- For fetching the full mini dataset: 
    python -m Common_voice_asr.fetch_mini --full_mini
- For processing the full mini dataset to transform the audio into mel spectograms: 
    python -m Common_voice_asr.preprocess --full_mini
- For training the RNN model using the full mini dataset for 5 epochs using CTCLoss:
    python -m neural_networks.modeling.train --full_mini --model_type rnn --epochs 5 --logdir runs/week4_ctc
- For training the CNN model using the full mini dataset for 5 epochs using CTCLoss:
    python -m neural_networks.modeling.train --full_mini --model_type cnn --epochs 5 --lr 1e-3 --logdir runs/week4_ctc
- For visualizing the loss & WER metrics documented with each training session:
    tensorboard --logdir Common-voice-asr/neural_networks/runs/week4_ctc

# Week 5: Hyperparameter Sweep
1. Define grid in `configs/week5_sweep.yaml`
    - Static parameters can be added as a list with only a single value
    - Method can be bayes, grid, or random. Began with Random for tests then moved to Bayes for fine-tuning on a larger scale
2. Run neural_networks/sweep.py in terminal:
    - python neural_networks/sweep.py
    - Does not require creating sweep or wand agent, it is handled in the code
        - Although you should create & log into an account with W&B
3. Analyze using the W&B's graphing & sorting on their website
4. Retrain best config: see models/best_cnn.pth or models/best_rnn.pth
    - Can also use log_best.py but replace hyperparameter inputs to train to that of your own best runs