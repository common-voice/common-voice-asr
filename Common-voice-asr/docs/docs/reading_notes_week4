# PyTorch CTCLoss documentation
CTC loss
- Calculates loss between a continuous (unsegmented) time series and a target sequence
- alignment of input to target is assumed to be “many-to-one”, which limits the length of the target sequence <= input length
Shape:
- Log_probs: Tensor of size (T,N,C) where T = input length, N = batch size, & C = # classes. Log probability of outputs
- Targets: Tensor of size (N, S) where S = max target lengths. Represents target sequences, each element is a class index that cannot be blank
- Inputs_lengths: tuple/sensor of size (N) or (). Represents lengths of inputs, must be <= T. Lengths specified for each sequence to achieve masking
- Target_lengths: Tuple or tensor of size (N) or (). Represents lengths of targets, specified for each sequence to achieve masking
- Output: scalar if reduction is 'mean' (default) or 'sum'. (N) if input is batched or () if input is unbatched if reduction is 'none'

# Custom collate_fn in PyTorch
- custom dataset should inherit Dataset and override the following methods:
    - __len__ so that len(dataset) returns the size of the dataset. read csv
    - __getitem__ to support indexing such that dataset[i] can be used to get ith sample. reads the spects
- custom transformations - preprocessing to fulfill expectation of fixed data size
    - callable classes rescale, randomcrop, totensor
- everytime dataset is sampled:
    - data is read from file on fly
    - transforms applied on the read data
    - Since one of the transforms is random, data is augmentated on sampling
    - collate_fn -- specifies exactly how samples need to be batched
# ASR Inference with CTC Decoder (torch.audio tutorial)
- Beam search decoding
    - iteratively expands text hypotheses (beams) w next possible characters, maintaining only the hypotheses with the highest scores at each time step
    - language model can be incorporated into scoring computation
        - adding a lexicon constraint restricts the next possible tokens for the hypotheses so that only words from the lexicon can be generated
- Running ASR inference using a CTC Beam Search decoder with a language model and lexicon constraint requires:
    - Acoustic Model: model predicting phonetics from audio waveforms
    - Tokens: the possible predicted tokens from the acoustic model
        - can either be passed in as a file, where each line consists of the tokens corresponding to the same index, or as a list of tokens, each mapping to a unique index
    - Lexicon: mapping between possible words and their corresponding tokens sequence
        - used to restrict the search space of the decoder to only words from the lexicon
        - expected format of the lexicon file is a line per word, w word followed by space-split tokens
    - Language Model (LM): n-gram language model trained w KenLM lib or custom lang model inheriting CTCDecoderLM
        - can be used in decoding to improve the results, by factoring in a language model score that represents the likelihood of the sequence into the beam search computation
Language Model
- None = set lm=None when initializing the decoder
- Custom Language Model
    - define using using CTCDecoderLM and CTCDecoderLMState
Greedy Decoder
- __init__ & forward
-   greedy_result = greedy_decoder(emission[0])
    greedy_transcript = " ".join(greedy_result)
    greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)
- can predict incorrectly spelled words like “affrayd” and “shoktd”

