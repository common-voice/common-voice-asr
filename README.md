# common-voice-asr
Constructing different types of neural networks trained on Common Voice speech datasets to compare the performance of approaches on speech to text performance metrics. 

# Quick-Start
1. Clone the repositiory
    git clone https://github.com/common-voice/common-voice-asr.git
    cd common-voice-asr/Common-voice-asr
2. Create the environment
    make create_environment
    conda activate $(PROJECT_NAME)
3. Install dependencies
    make requirements
4. Train the model
    make train
5. Test the model
    make test