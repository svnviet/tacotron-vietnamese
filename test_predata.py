from util import audio
import numpy as np
import librosa
import os

wav_t = 'LJSpeech-1.1/wavs/audios/Audio_1280_2560.wav'
out_dir = 'test_predata'

wav = audio.load_wav(wav_t)
spectrogram = audio.spectrogram(wav).astype(np.float32)
n_frames = spectrogram.shape[1]
mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

np.save(os.path.join(out_dir, 'spectrogram_filename'), spectrogram.T, allow_pickle = False)
np.save(os.path.join(out_dir, 'mel_spectrogram_filename'), mel_spectrogram.T, allow_pickle = False)
