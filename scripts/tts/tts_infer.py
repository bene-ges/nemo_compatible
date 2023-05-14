from argparse import ArgumentParser

import soundfile as sf
import torch

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

parser = ArgumentParser(description="Run TTS")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_dir", type=str, required=True, help="Output dir with .wav files")
parser.add_argument("--output_manifest", type=str, required=True, help="Output manifest file")
parser.add_argument("--spec_generator", type=str, required=True, help="Path to .nemo checkpoint of spectrogram generator, e.g. FastPitch.nemo")
parser.add_argument("--vocoder", type=str, required=True, help="Path to .nemo checkpoint of vocoder, e.g. HifiGan.nemo")
parser.add_argument("--sample_rate", type=int, default=22050, help="Output sample rate")

args = parser.parse_args()

# Download and load the pretrained fastpitch model
if torch.cuda.is_available():
    spec_generator = SpectrogramGenerator.restore_from(args.spec_generator).cuda()
else:
    spec_generator = SpectrogramGenerator.restore_from(args.spec_generator)
spec_generator.eval()

# Download and load the pretrained hifigan model
# vocoder = Vocoder.from_pretrained(model_name="tts_en_hifigan").cuda()
if torch.cuda.is_available():
    vocoder = Vocoder.restore_from(args.vocoder).cuda()
else:
    vocoder = Vocoder.restore_from(args.vocoder)


out_manifest = open(args.output_manifest, "w", encoding="utf-8")

lid = 0
with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        inp = line.strip()

        # Produce a spectrogram
        spectrogram = spec_generator.generate_spectrogram(tokens=spec_generator.parse(inp))

        # Finally, a vocoder converts the spectrogram to audio
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        # Note that vocoder return a batch of audio. In this example, we just take the first and only sample.
        filename = args.output_dir + "/" + str(lid) + ".wav"
        sf.write(filename, audio.to('cpu').detach().numpy()[0], args.sample_rate)
        # {"audio_filepath": "tts/1.wav", "text": "ndimbati"}
        out_manifest.write(
            "{\"audio_filepath\": \"" + filename + "\", \"text\": \"" + inp + "\"}\n"
        )
        lid += 1

out_manifest.close()
