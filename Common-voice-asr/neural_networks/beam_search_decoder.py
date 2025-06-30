# https://docs.pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
# https://pytorch.org/blog/fast-beam-search-decoding-in-pytorch-with-torchaudio-and-flashlight-text/
# python -m neural_networks.modeling.train --full_mini --model_type rnn --epochs 5 --lr 1e-3 --logdir runs/week6_beam 
# python -m neural_networks.modeling.train --full_mini --model_type rnn --epochs 5 --lr 1e-3 --logdir runs/week6_beam --lm-weight 0 --word-score 0
import os
from torchaudio.models.decoder import ctc_decoder
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))
LEXICON_PATH = BASE_DIR / "data" / "lexicon.txt"
LM_PATH = BASE_DIR / "data" / "cleaned_lm.bin"

def beam_search_decoder(tokens, lm_weight, word_score):
    decoder = ctc_decoder(str(LEXICON_PATH), tokens, str(LM_PATH), nbest=3, beam_size=100, lm_weight=lm_weight, word_score=word_score)
    def decode_batch(log_probs_batch):
        decoded = decoder(log_probs_batch.cpu())
        return [hyp[0].words for hyp in decoded]
    return decode_batch