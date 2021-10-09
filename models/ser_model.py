"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import  Wav2Vec2Model
from models.ser_spec import SER_AlexNet


# __all__ = ['Ser_Model']
class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        
        # CNN for Spectrogram
        self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        
        # LSTM for MFCC        
        self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        
        # Spectrogram + MFCC  
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
                        
        # WAV2VEC 2.0
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/heqing001/Coding/SER_0915/features_extraction/pretrained_model/wav2vec2-base-960h")

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128) # 512 for 1 and 768 for 2
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)
        
                                                                     
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # spectrogram - SER_CNN
        audio_spec, output_spec_t = self.alexnet_model(audio_spec) # [batch, 256, 6, 6], []
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1) # [batch, 256, 36]  
        
        # audio -- MFCC with BiLSTM
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 300, 512]  
        
        audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 9216]  
        audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 9216]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  
        
        #+ audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 153600]  
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 153600]  
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  
        

        # FOR WAV2VEC2.0 WEIGHTS 
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1) # [batch, 256] 
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)# [batch, 256] 
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=False)# [batch, 149] 
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)# [batch, 1, 149] 
        #+ audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)

        # wav2vec 2.0 
        audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 768] 
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768] 
        #audio_wav = torch.mean(audio_wav, dim=1)
        
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 768] 
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False) # [batch, 768] 
        
        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] 
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
        
  
        output = {
            'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        

        return output
    
