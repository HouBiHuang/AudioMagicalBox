import librosa
#from SpecAugment import spec_augment_tensorflow
import SpecAugment.spec_augment_tensorflow
import numpy as np
from scipy.io import wavfile
import importlib
importlib.reload(SpecAugment.spec_augment_tensorflow)

words = ["ㄏㄧㄡ"]#,"ㄟ","他","好","你","吼","我","那","那那個","的","的一個","的那個","的這個","阿","啦","著","嗯"]

if __name__ == "__main__":
    for word in words:
        for i in range(1,2):
            audio_path = "./recordingSpecAugment/{0}/{0}{1}.wav".format(word,i) #args.audio_path
            # Step 0 : load audio file, extract mel spectrogram
            audio, sampling_rate = librosa.load(audio_path,sr=16000)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                             sr=sampling_rate,
                                                             n_mels=256,
                                                             fmax=16000)
            
            # Show Raw mel-spectrogram
            SpecAugment.spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
                                                              title="Raw Mel Spectrogram")
            
            for j in range(1,6):
                
                #做資料增強
                warped_masked_spectrogram = SpecAugment.spec_augment_tensorflow.spec_augment(mel_spectrogram)
            
                # Show time warped & masked spectrogram
                SpecAugment.spec_augment_tensorflow.visualization_tensor_spectrogram(mel_spectrogram=warped_masked_spectrogram,
                                                                  title="tensorflow Warped & Masked Mel Spectrogram")
                
                
                #Change spectrogram to audio
                mel_to_stft = librosa.feature.inverse.mel_to_stft(warped_masked_spectrogram)
                output_audio = librosa.griffinlim(mel_to_stft)
                
                #因轉成audio的長度不夠，所以用0填充到8000行
                window = np.zeros(8000)
                output_audio_len = len(output_audio)
                window[0:output_audio_len] = output_audio
                
                #Output wav
                wavfile.write("./recordingSpecAugment/{0}/warped_time_frequency_{0}{1}-{2}.wav".format(word,i,j), 16000, window)    
                print("{0}-{1}-{2}".format(word,i,j))
                
    