import librosa
from specAugment import spec_augment_tensorflow
# If you are Pytorch, then import spec_augment_pytorch instead of spec_augment_tensorflow
audio, sampling_rate = librosa.load("./ㄏㄧㄡ1.wav",sr=16000)
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)
warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)
print(warped_masked_spectrogram)