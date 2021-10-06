import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import numpy as np

audio_path = "./ㄏㄧㄡ1.wav" #args.audio_path
audio, sampling_rate = librosa.load(audio_path,sr=16000)


mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=sampling_rate,
                                                 n_mels=256,
                                                 fmax=16000)

shape = mel_spectrogram.shape
mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sampling_rate,
                         fmax=16000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')





