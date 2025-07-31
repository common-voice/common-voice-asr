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
    
# Final Instructions:
Set Up Env & Directory to Operate From:
- 
     make create_environment
     conda activate Common-voice-asr
     cd common-voice-asr
     pip install -r requirements.txt
     cd Common-voice-asr
     
     * Unless otherwise specified, run all further commands from Common-voice-asr
Data & Preprocessing:
- 
1. Fetch: Grab data from Common Voice corpus folder and create CSVs with filename, transcript, & duration. Seperated into Train & Validation folders

        python -m fetch_data --corpus
        
    * Note: if the folder is outside of the main common-voice-asr directory, run this command from the base:

            python -m common-voice-asr.Common-voice-asr.fetch_data --corpus
            
    * You will also need to update DATA_DIR_CORPUS within fetch_data.py to specify the name of the corpus folder to retrieve from
    
2. Preprocessing: Process audio files into mel-spectograms
    
        python -m preprocess --corpus
        
    * If you want to test mel spectogram generation to find the best build parameters, you can change the parameter lists & run:
    
            python -m preprocess --sample
            
    * Then, once you have the best parameters:
            
            python -m preprocess --corpus --best
            
    * Note: Once again, if you have to run from the base the command will be:
            
            python -m common-voice-asr.Common-voice-asr.preprocess --corpus

Testing:
* Run from common-voice-asr (if you are in Common-voice-asr step backward with "cd ..")
    
        pytest

Training:
- Basic Setup:

    python -m neural_networks.modeling.train --model [rnn, cnn, or transformer] --logdir runs/[name] --corpus 
    
- Additional commands:
    * Check how data loads, will return without starting training: --check_data 
    * Specify number of epochs: -- epochs [int]
    * Learning rate: --lr [float]
    * Use greedy decoding instead of beam search: --greedy
    * Batch size: --batch-size [int]
    * Sample only a part of the corpus: --sample_size [int] (must be less than total length & able to split train and dev into 85% & 15% respectively without exceeding their lengths)
    * --debug_sample (run only one sample for a single training loop)
    * Model parameters:
        - --hidden_dim [int]
        - --dropout [float]
        Model B specific parameters
        - --d_model [int] (must be an even number to avoid errors)
        - --nhead [int]
        - --dim_feedforward [int]
        - --nlayers [int]
        - --lstm_hidden [int]
        - --lstm_layers [int]
        - --conv_layer (add a convolutional layer to model)
    * Beam search parameters:
        - --lm_weight [float]
        - --word_score [float]
        - --beam_width [int]
    Full example: 
            
            python -m neural_networks.modeling.train   --corpus   --model_type transformer   --epochs 15   --dropout 0.1   --lr 1e-3   --logdir runs/tuning_transformer   --batch-size 16   --greedy  --d_model 256 --dim_feedforward 512 --nlayers 2

Sweeps:
- Constructing sweep YAML:
    * Use the example configurations to guide your construction, or use the ones already available. The project name should match what you pass into the command line.
    * Store under neural_networks/configs
    * Main components:
        - Method: defines how the hyperparameter sweep will conduct its exploration (e.g. bayes, random, grid)
        - Metric: the value that you want to guide hyperparameter exploration, specifying a name & goal. In the case of our ASR, the goal is to minimize validation WER
        - Parameters
            * Must specify logdir and model, as these are not specific to your run in the default YAML. If you are sweeping model a: rnn, model b: transformer
- WandB sweep:
    * Create & log into an account with W&B before running the sweep. If it prompts you to login through the terminal, you may have to create an API key, which can be handled on WandB's website.
    * Run the sweep:

            python neural_networks/sweep.py --model [a or b] --config [just the YAML file] --project_name [str]
            ex. python neural_networks/sweep.py --model a --config week8_sweep_a.yaml --project_name "week8_sweep_a" --count 20

Troubleshooting:
- 
- General:

- Model A:

- Model B:

Helpful commands & keys:
- Ctrl + C to stop the run in the command line
- Use up & down arrows to navigate prior commands

        
