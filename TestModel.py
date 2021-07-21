import sounddevice as sd
import numpy as np
import scipy.signal
import python_speech_features
import tensorflow as tf
from PyQt5.QtCore import QTime
import librosa

# This gets called every 0.5 seconds
def sound(window,s,m):
    
    S = np.abs(librosa.stft(window)) #將整個window音訊做stft，並轉成絕對值
    
    if np.sum(S) > 3000: #判斷S的總和是否>3000，如果>3000，代表有講話
        # Compute features
        mfccs = python_speech_features.base.mfcc(window, #輸入訊號
                                            samplerate=8000, #取樣率
                                            winlen=0.256, #音框涵蓋時間
                                            winstep=0.050, #音框間距離
                                            numcep=num_mfcc, #返回係數的量
                                            nfilt=26, #過濾器數量
                                            nfft=2048,#FFT大小
                                            preemph=0.0,#不用預強化濾波器
                                            ceplifter=0,#ROBUST
                                            appendEnergy=True,#係數0的話對被替代成總音框能量的對數
                                            winfunc=np.hanning)#hanning window
        mfccs = mfccs.transpose()
        
        # Make prediction from model
        in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
        #設定輸入張量
        interpreter.set_tensor(input_details[0]['index'], in_tensor)
        #進行預測
        interpreter.invoke()
        #取得輸出張量
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        val = output_data[0]#取得預測值
        val = val.tolist() #np.ndarray to list
        list_val_max = max(val) #取得最大值
        list_val_maxIndex = val.index(max(val)) #取得最大值的索引  
        
        if list_val_maxIndex == 2:
            data.append(str(m) + "分" + str(s) + "秒，出現:" + str(words[list_val_maxIndex]) + "，預測值:" + str(list_val_max))
         
#main
# Parameters
num_mfcc = 23 #回傳mfcc的量
model_path = './recording23.tflite'
words = ['ㄏㄧㄡ', 'ㄟ', '吼', '啦', '嗯', '的一個', '的這個', '的那個', '著', '那', '那那個', '阿']#答案對應到的字詞
data = []
start = 0 #一開始的索引值
end = 4000 #一開始的索引值
s = 0.5 #秒
m = 0 #分
duration = 180 #讀音檔的總時間
sample_rate = 8000 #取樣率

#載入音檔
y, sr = librosa.load("./chen.wav",sr=sample_rate,duration=duration) 

# Sliding window
window = np.zeros(8000)#取樣音頻數據變數

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


while True:
    window[0:4000] = y[start:end] #把音訊載入window
    start = start + 4000 #向後移動
    end = end + 4000 #向後移動
    window[4000:] = y[start:end] #把音訊載入window
    s = s + 0.5 #增加秒數
    if(s == 60): #60秒 轉成 1分
        s = 0
        m = m + 1
    sound(window,s,m) #呼叫sound()
    if(end == 1440000): #如果移動到最後，break
        break
    
for i in range(len(data)):
        print(data[i])
    
    
    
    