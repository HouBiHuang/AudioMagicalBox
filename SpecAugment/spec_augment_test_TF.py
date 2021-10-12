import argparse
import librosa
from SpecAugment import spec_augment_tensorflow
import os, sys
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default='../data/61-70968-0002.wav',
                    help='The audio file.')
parser.add_argument('--time-warp-para', default=80,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=27,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=100,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number

words = ["ㄏㄧㄡ","ㄟ","他","好","你","吼","我","那","那那個","的","的一個","的那個","的這個","阿","啦","著","嗯"]

if __name__ == "__main__":
    for word in words:
        for i in range(1,41):
            audio_path = "./recordingSpecAugment/{0}/{0}{1}.wav".format(word,i) #args.audio_path
            # Step 0 : load audio file, extract mel spectrogram
            audio, sampling_rate = librosa.load(audio_path,sr=16000)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                             sr=sampling_rate,
                                                             n_mels=256,
                                                             fmax=16000)
            
            # reshape spectrogram shape to [batch_size, time, frequency, 1]
            shape = mel_spectrogram.shape
            #mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))
            # Show Raw mel-spectrogram
            #spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
            #                                                  title="Raw Mel Spectrogram")
            
            for j in range(1,6):
                
                #做資料增強
                warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram)
            
                # Show time warped & masked spectrogram
                #spec_augment_tensorflow.visualization_tensor_spectrogram(mel_spectrogram=warped_masked_spectrogram,
                #                                                  title="tensorflow Warped & Masked Mel Spectrogram")
                
                #print(warped_masked_spectrogram.shape)
                
                #warped_masked_spectrogram = np.reshape(warped_masked_spectrogram, (shape[0], shape[1]))
                #print(warped_masked_spectrogram)
                
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
                
    