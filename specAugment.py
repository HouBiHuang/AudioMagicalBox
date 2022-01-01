import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile
import random

#音訊加噪
def noise_augment(data, noise_factor = 0.05):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

#平移時間
def shifting_time(data, sampling_rate = 16000, shift_max = 0.1, shift_direction = 'both'):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

#改變音高
def changing_pitch(data, sampling_rate = 16000):
    bins_per_octave = np.random.randint(-5,12)
    #print(bins_per_octave)
    return librosa.effects.pitch_shift(data, sampling_rate, bins_per_octave)

#改變速度
def _aug_speed(sig):
    speed_rate = random.Random().uniform(1.1, 1.3)
    old_length = sig.shape[0]
    new_length = int(sig.shape[0] / speed_rate)
    old_indices = np.arange(old_length)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)
    sig = np.interp(new_indices, old_indices, sig)
    return sig

words = ["吼"]
for word in words:
    for i in range(1,2): #讀取每個資料夾下的每個檔案
        audio_path = "./吼{0}.wav".format(i)
            
        audio, sampling_rate = librosa.load(audio_path,sr=16000)       
        librosa.display.waveplot(audio, sr=16000)
        plt.tight_layout()
        plt.show()
        
        for j in range(1,2): #每個檔案產生j個資料增強後的檔案
            audio_augment = changing_pitch(audio)
            
            #做改變速度增強時，因改變後會不足0.5s，所以用0補齊
            #audio_augment_len = len(audio_augment)
            #output_audio = np.zeros(8000)   
            #output_audio[:audio_augment_len] = audio_augment
            
            librosa.display.waveplot(audio_augment, sr=16000)
            plt.tight_layout()
            plt.show()

            #Output wav
            wavfile.write("./TEST.wav", 16000, audio_augment)
                
            #print("{0}_3_{1}_noise_shifting-time_changing-pitch.wav".format(i,j))

