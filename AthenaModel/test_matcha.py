from inference import loadTTS_model, load_vocoder, process_text, _MATCHA_CHECKPOINT, _HIFIGAN_CHECKPOINT

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

from time import time
from inference import synthesise, to_waveform, save_to_folder
import torch
import datetime as dt

text = "testing"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#TTS
ttsmodel = MatchaTTS.load_from_checkpoint(_MATCHA_CHECKPOINT, map_location=device)
ttsmodel.eval()

#Vocoder
h = AttrDict(v1)
hifigan = HiFiGAN(h).to(device)
hifigan.load_state_dict(torch.load(_HIFIGAN_CHECKPOINT, map_location=device)['generator'])
_ = hifigan.eval()
hifigan.remove_weight_norm()
vocoder = hifigan

#denoiser
denoiser = Denoiser(vocoder, mode='zeros')

@torch.inference_mode()
def synthesise(text):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = ttsmodel.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=2,
        temperature=0.8,
        spks=torch.tensor([1], device=device, dtype=torch.long).to(device),
        length_scale=0.7 
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

startTTS = time()
output = synthesise(text) 
output['waveform'] = to_waveform(output['mel'], vocoder)
endTTS = time()
print("total amount of audio time : ", endTTS - startTTS)

startetc = time()
originalWav = output['waveform']
save_to_folder("input_audio", output, "./")