import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile

def noise_augment(data, noise_factor = 0.05):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

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

def changing_pitch(data, sampling_rate = 16000):
    bins_per_octave = np.random.randint(-5,12)
    #print(bins_per_octave)
    return librosa.effects.pitch_shift(data, sampling_rate, bins_per_octave)

#main
words = ["ㄏㄧㄡ","ㄟ","他","好","你","吼","我","那","那那個","的","的一個","的那個","的這個","阿","啦","著","嗯"]
for word in words:
    for i in range(1,41):
        audio_path = "./recording2/{0}/{0}{1}.wav".format(word,i)
            
        audio, sampling_rate = librosa.load(audio_path,sr=16000)       
        #librosa.display.waveplot(audio, sr=16000)
        #plt.tight_layout()
        #plt.show()
        
        for j in range(1,6):
            audio_augment1 = noise_augment(audio)
            audio_augment2 = shifting_time(audio_augment1)
            audio_augment3 = changing_pitch(audio_augment2)
            
            #librosa.display.waveplot(audio_augment3, sr=16000)
            #plt.tight_layout()
            #plt.show()

            wavfile.write("./recording2/{0}/noise_shifting-time_changing-pitch_{0}{1}-{2}.wav".format(word,i,j), 16000, audio_augment3)    
            print("{0}-{1}-{2}".format(word,i,j))

