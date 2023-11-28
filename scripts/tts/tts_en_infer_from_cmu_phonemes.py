"""
This script allows to run TTS inference directly from CMU phonemes.

python ${NEMO_COMPATIBLE_PATH}/scripts/tts/tts_en_infer_from_cmu_phonemes.py \
  --input_name input.txt \
  --output_dir out \
  --output_manifest out.json

Input file is expected to be in two-column format: 
    original text
    comma-separated phoneme sequence
Input line example:
    jocelyn geffert \t JH,AO1,S,L,IH0,N, ,G,EH1,F,ER0,T 

Note that changing spec_generator and vocoder to other than default options has not been checked and is not recommended.
If you plan to run ASR after TTS, consider changing sample_rate parameter depending on what ASR expects, e.g. 16000.

"""

from argparse import ArgumentParser

import soundfile as sf
import torch

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.collections.tts.torch.g2ps import EnglishG2p
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishPhonemesTokenizer

parser = ArgumentParser(description="Run TTS inference directly from CMU phonemes")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_dir", type=str, required=True, help="Output dir with .wav files")
parser.add_argument("--output_manifest", type=str, required=True, help="Output manifest file")
parser.add_argument("--spec_generator", type=str, default="tts_en_fastpitch", help="Name of pretrained spectrogram generator")
parser.add_argument("--vocoder", type=str, default="tts_en_hifigan", help="Name of pretrained vocoder")
parser.add_argument("--sample_rate", type=int, default=22050, help="Output sample rate")

args = parser.parse_args()

# Download and load the pretrained fastpitch model
spec_generator = SpectrogramGenerator.from_pretrained(model_name=args.spec_generator).cuda()
spec_generator.eval()

# Download and load the pretrained hifigan model
vocoder = Vocoder.from_pretrained(model_name=args.vocoder).cuda()

# Here EnglishG2p is needed only to initialize.
# After that .encode_from_g2p accepts phonemes directly.
text_tokenizer = EnglishPhonemesTokenizer(
    punct=True, stresses=True, chars=True, space=' ', apostrophe=True, pad_with_space=True, g2p=EnglishG2p(),
)

out_manifest = open(args.output_manifest, "w", encoding="utf-8")

lid = 0
with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        # jocelyn geffert \t JH,AO1,S,L,IH0,N, ,G,EH1,F,ER0,T 
        line = line.strip()
        try:
            raw, inp = line.split("\t")
        except:
            print("bad format:", line)
            continue
        
        # arg: list of phonemes e.g. ["AA1", "M", "AH0"]
        parsed = text_tokenizer.encode_from_g2p(inp.split(","))

        parsed = torch.Tensor(parsed).to(dtype=torch.int64, device=spec_generator.device)
        parsed = torch.unsqueeze(parsed, 0)

        # Make spectrogram
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)

        # Converts the spectrogram to .wav
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        # Save the audio to disk
        # Note that vocoder return a batch of audio. In this example, we just take the first and only sample.
        filename = args.output_dir + "/" + str(lid) + ".wav"
        sf.write(filename, audio.to('cpu').detach().numpy()[0], args.sample_rate)
        # {"audio_filepath": "out/0.wav", "text": "jocelyn geffert", "g2p": "JH,AO1,S,L,IH0,N, ,G,EH1,F,ER0,T"}
        out_manifest.write(
            "{\"audio_filepath\": \"" + filename + "\", \"text\": \"" + raw + "\", \"g2p\": \"" + inp + "\"}\n"
        )
        lid += 1
out_manifest.close()
