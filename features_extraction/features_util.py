import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
from tqdm import tqdm
from pysndfx import  AudioEffectsChain
from create_mixed_audio_file import create_mixed_audio_file
import random
from transformers import BertTokenizer, BertModel, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, AutoTokenizer

NOISE_DIR='noise_wav/'
NOISE_FILES={'cafetaria.wav': 0,
             #'car.wav': 0,
             'kitchen.wav': 0,
             'park.wav': 6,
             'traffic.wav': 6,
             #'metro.wav': 0,
             'presto.wav': 10,
             #'station.wav': 6,
             'river.wav':0,
             'square.wav':0
             }
             
def extract_features(speaker_files, features, params):
    
    processor = Wav2Vec2Processor.from_pretrained("/home/heqing001/Coding/SER_0915/features_extraction/pretrained_model/wav2vec2-base-960h")
    speaker_features = defaultdict()
    # data_mfcc = list()
    for speaker_id in tqdm(speaker_files.keys()):
        
        data_tot, labels_tot, labels_segs_tot, segs, data_mfcc, data_audio = list(), list(), list(), list(), list(), list()
        for wav_path, emotion in speaker_files[speaker_id]:
            
            # Read wave data
            x, sr = librosa.load(wav_path, sr=None)

            # Apply pre-emphasis filter
            x = librosa.effects.preemphasis(x, zi = [0.0])

            # Extract required features into (C,F,T)
            features_data = GET_FEATURES[features](x, sr, params)
            
            hop_length = 160 # hop_length smaller, seq_len larger
            # f0 = librosa.feature.zero_crossing_rate(x, hop_length=hop_length).T # (seq_len, 1)
            # cqt = librosa.feature.chroma_cqt(y=x, sr=sr, n_chroma=24, bins_per_octave=72, hop_length=hop_length).T # (seq_len, 12)
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T # (seq_len, 20)
            
            # wav2vec
            # input_values = processor(x, sampling_rate=sr, return_tensors="pt").input_values
            
            # Segment features into (N,C,F,T)
            features_segmented = segment_nd_features(x, mfcc, features_data, emotion, params['segment_size'])

            #Collect all the segments
            data_tot.append(features_segmented[1])
            labels_tot.append(features_segmented[3])
            labels_segs_tot.extend(features_segmented[2])
            segs.append(features_segmented[0])
            data_mfcc.append(features_segmented[4])
            data_audio.append(features_segmented[5])

        # Post process
        data_tot = np.vstack(data_tot).astype(np.float32)
        data_mfcc = np.vstack(data_mfcc).astype(np.float32)
        data_audio = np.vstack(data_audio).astype(np.float32)
        labels_tot = np.asarray(labels_tot, dtype=np.int8)
        labels_segs_tot = np.asarray(labels_segs_tot, dtype=np.int8)
        segs = np.asarray(segs, dtype=np.int8)
        
        # Make sure everything is extracted properly
        assert len(labels_tot) == len(segs)#+ == data_mfcc.shape[0]
        assert data_tot.shape[0] == labels_segs_tot.shape[0] == sum(segs)

        # Mix with noise
        if params['mixnoise'] == True:
            data_tot = np.expand_dims(data_tot, axis = 0)
            
            # + NOISE
            for noise in NOISE_FILES.keys():
                data_noise_tot = list()
                noise_path = NOISE_DIR+noise
                labels_temp, segs_temp = list(), list() #for verification
                for wav_path, emotion in speaker_files[speaker_id]:
                    create_mixed_audio_file(wav_path, noise_path, 
                                            'temp_noise.wav', NOISE_FILES[noise])
                
                    # Load mixed speech
                    x_noise, sr_noise = librosa.load('temp_noise.wav', sr=None)
                    assert sr_noise == sr
                
                    # Pre-emphasis
                    x_noise = librosa.effects.preemphasis(x_noise, zi = [0.0])
                    #assert len(x_noise) == len(x)
                
                    # Extract required features into (C,F,T)
                    features_data_noise = GET_FEATURES[features](x_noise, sr, params)
                    
                    hop_length = 160 # hop_length smaller, seq_len larger
                    # f0 = librosa.feature.zero_crossing_rate(x, hop_length=hop_length).T # (seq_len, 1)
                    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T # (seq_len, 20)
                    
                    input_values = processor(x, sampling_rate=sr, return_tensors="pt").input_values
                    
                    # Segment features into (N,C,F,T)
                    features_noise_segmented = segment_nd_features(segment_nd_features, mfcc, features_data_noise, emotion, params['segment_size'])

                    #Collect all the segments
                    data_noise_tot.append(features_noise_segmented[1])
                    labels_temp.append(features_noise_segmented[3])
                    segs_temp.append(features_noise_segmented[0])

                    os.remove('temp_noise.wav')
                
                data_noise_tot = np.vstack(data_noise_tot).astype(np.float32)
                data_noise_tot = np.expand_dims(data_noise_tot, axis=0)
                assert data_noise_tot.shape[1] == data_tot.shape[1]
                
                labels_temp = np.asarray(labels_temp, dtype=np.int8)
                segs_temp = np.asarray(segs, dtype=np.int8)
                assert np.array_equal(labels_temp, labels_tot)
                assert np.array_equal(segs_temp, segs)
                
                data_tot = np.concatenate((data_tot, data_noise_tot), axis=0)

            # + REVERB
            data_reverb_tot = list() 
            labels_temp, segs_temp = list(), list() #for verification
            fx = (AudioEffectsChain().reverb())
            
            for wav_path, emotion in speaker_files[speaker_id]:
                    
                # Read wave data
                x, sr = librosa.load(wav_path, sr=None)
                # Apply reverb
                x_reverb = fx(x)
                    
                # Pre-emphasis
                x_reverb = librosa.effects.preemphasis(x_reverb, zi = [0.0])
                    
                # Extract required features into (C,F,T)
                features_data_reverb = GET_FEATURES[features](x_reverb, sr, params)

                hop_length = 160 # hop_length smaller, seq_len larger
                # f0 = librosa.feature.zero_crossing_rate(x, hop_length=hop_length).T # (seq_len, 1)
                mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T # (seq_len, 20)
                
                input_values = processor(x, sampling_rate=sr, return_tensors="pt").input_values
                
                # Segment features into (N,C,F,T)
                features_reverb_segmented = segment_nd_features(input_values, mfcc, features_data_reverb, emotion, params['segment_size'])

                #Collect all the segments
                data_reverb_tot.append(features_reverb_segmented[1])
                labels_temp.append(features_reverb_segmented[3])
                segs_temp.append(features_reverb_segmented[0])

            data_reverb_tot = np.vstack(data_reverb_tot).astype(np.float32)
            data_reverb_tot = np.expand_dims(data_reverb_tot, axis=0)
            assert data_reverb_tot.shape[1] == data_tot.shape[1]
                
            labels_temp = np.asarray(labels_temp, dtype=np.int8)
            segs_temp = np.asarray(segs, dtype=np.int8)
            assert np.array_equal(labels_temp, labels_tot)
            assert np.array_equal(segs_temp, segs)
                
            data_tot = np.concatenate((data_tot, data_reverb_tot), axis=0)

            # Make sure everything is extracted properly
            assert data_tot.shape[0] == len(NOISE_FILES.keys())+2

        #Put into speaker features dictionary
        print(data_tot.shape)
        print(labels_segs_tot.shape)
        print(data_audio.shape)
        print(labels_tot.shape)
        print(segs.shape)
        print(labels_tot.shape)
        audio_features = defaultdict()
        audio_features["seg_spec"] = data_tot
        audio_features["utter_label"] = labels_tot
        audio_features["seg_label"] = labels_segs_tot
        audio_features["seg_num"] = segs
        audio_features["seg_mfcc"] = data_mfcc
        audio_features["seg_audio"] = data_audio
        speaker_features[speaker_id] = audio_features #(data_tot, labels_tot, labels_segs_tot, segs)
        #if params['mixnoise'] == True:
        #    speaker_features[speaker_id] = ((data_tot,data_noise_tot), labels_tot, labels_segs_tot, segs )
        #else:
        #    speaker_features[speaker_id] = (data_tot, labels_tot, labels_segs_tot, segs)
    '''
    utter_begin = 0
    utter_end = 0
    for speaker_id in tqdm(speaker_files.keys()):
        audio_features = defaultdict()
        audio_features["seg_spec"] = speaker_features[speaker_id]["seg_spec"] 
        audio_features["utter_label"] = speaker_features[speaker_id]["utter_label"] 
        audio_features["seg_label"] = speaker_features[speaker_id]["seg_label"] 
        audio_features["seg_num"] = speaker_features[speaker_id]["seg_num"] 
        
        data_mfcc = paddingSequence(data_mfcc)
        data_mfcc = np.asarray(data_mfcc, dtype=np.float32)
        
        utter_end += audio_features["utter_label"].shape[0]
        audio_features["utter_mfcc"] = data_mfcc[utter_begin:utter_end]
        utter_begin += audio_features["utter_label"].shape[0]
        
        speaker_features[speaker_id] = audio_features
    '''
    
    assert len(speaker_features) == len (speaker_files)

    return speaker_features

def padding(feature, MAX_LEN):
    """
    mode: 
        zero: padding with 0
        normal: padding with normal distribution
    location: front / back
    """
    padding_mode  = 'zeros'
    padding_location = 'back'

    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :]
        
    if padding_mode == "zeros":
        pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
    elif padding_mode == "normal":
        mean, std = feature.mean(), feature.std()
        pad = np.random.normal(mean, std, (MAX_LEN-length, feature.shape[1]))

    feature = np.concatenate([pad, feature], axis=0) if(padding_location == "front") else \
              np.concatenate((feature, pad), axis=0)
    return feature
                  
def paddingSequence(sequences):
    if len(sequences) == 0:
        return sequences
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    # confirm length using (mean + std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))
    # padding sequences to final_length
    final_sequence = np.zeros([len(sequences), final_length, feature_dim])
    for i, s in enumerate(sequences):
        final_sequence[i] = padding(s, final_length)

    return final_sequence
        
def extract_logspec(x, sr, params):
    
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:nfreq]
    
    #Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec,0)

    return spec


def extract_lognrevspec(x, sr, params):
    
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    #Input + gaussian white noise
    noise=np.random.normal(0, 0.002, x.shape[0])
    x_noisy = x+noise
    
    #Input + reverb
    fx = (AudioEffectsChain().reverb())
    x_reverb = fx(x)
    
    #calculate stft
    spec = np.abs(librosa.stft(x, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)

    spec_noise = np.abs(librosa.stft(x_noisy, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec_noise =  librosa.amplitude_to_db(spec_noise, ref=np.max)

    spec_reverb = np.abs(librosa.stft(x_reverb, n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window))
    
    spec_reverb =  librosa.amplitude_to_db(spec_reverb, ref=np.max)

    # Get the required frequency bins
    spec = spec[:nfreq]
    spec_noise = spec_noise[:nfreq]
    spec_reverb = spec_reverb[:nfreq]
    
    # Arrange into (C, F, T), C = 3
    spec = np.expand_dims(spec,0)
    spec_noise = np.expand_dims(spec_noise,0)
    spec_reverb = np.expand_dims(spec_reverb,0)
    spec = np.concatenate((spec, spec_noise, spec_reverb), axis=0)

    return spec



def extract_logmelspec(x, sr, params):
 
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    

    #calculate stft
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                        n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window)
    
    logmelspec =  librosa.power_to_db(melspec, ref=np.max)

    # Expand to (C, F, T), C = 3
    logmelspec =  np.expand_dims(logmelspec, 0)
    
    return logmelspec


def extract_logmeldeltaspec(x, sr, params):
      
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    
    #calculate stft
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                        n_fft=ndft,hop_length=hop_length,
                                        win_length=win_length,
                                        window=window)
    logmelspec =  librosa.power_to_db(melspec, ref=np.max)
    
    #logmeldeltaspec = librosa.feature.delta(logmelspec)
    #logmeldelta2spec = librosa.feature.delta(logmelspec, order=2)
    logmeldeltaspec = librosa.feature.delta(logmelspec,width=5,mode='nearest')
    logmeldelta2spec = librosa.feature.delta(logmeldeltaspec, width=5,mode='nearest')
    
    #Arrange into (C, F, T), C = 3
    logmelspec = np.expand_dims(logmelspec, axis=0)
    logmeldeltaspec = np.expand_dims(logmeldeltaspec, axis=0)
    logmeldelta2spec = np.expand_dims(logmeldelta2spec, axis=0)
    logmelspec = np.concatenate((logmelspec, logmeldeltaspec, logmeldelta2spec), axis=0)
    
    return logmelspec


def extract_logdeltaspec(x, sr, params):
      
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sr)
    hop_length    = int((params['hop_length']/1000) * sr)
    ndft          = params['ndft']
    n_freq        = params['nfreq']
    
    #calculate stft
    logspec = extract_logspec(x, sr, params) # (C, F, T)

    logdeltaspec = librosa.feature.delta(logspec.squeeze(0))
    logdelta2spec = librosa.feature.delta(logspec.squeeze(0), order=2)
    
    #Arrange into (C, F, T), C = 3
    logdeltaspec = np.expand_dims(logdeltaspec, axis=0)
    logdelta2spec = np.expand_dims(logdelta2spec, axis=0)
    logspec = np.concatenate((logspec, logdeltaspec, logdelta2spec), axis=0)
    
    return logspec


def segment_nd_features(input_values, mfcc, data, emotion, segment_size):
    '''
    Segment features into <segment_size> frames.
    Pad with 0 if data frames < segment_size

    Input:
    ------
        - data: shape is (Channels, Fime, Time)
        - emotion: emotion label for the current utterance data
        - segment_size: length of each segment
    
    Return:
    -------
    Tuples of (number of segments, frames, segment labels, utterance label)
        - frames: ndarray of shape (N, C, F, T)
                    - N: number of segments
                    - C: number of channels
                    - F: frequency index
                    - T: time index
        - segment labels: list of labels for each segments
                    - len(segment labels) == number of segments
    '''
    segment_size_wav = segment_size * 160
    # Transpose data to C, T, F
    
    data = data.transpose(0,2,1)
    time = data.shape[1]
    time_wav = input_values.shape[0]
    nch = data.shape[0]
    start, end = 0, segment_size
    start_wav, end_wav = 0, segment_size_wav
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    #if num_segs > 1:
    #    num_segs = num_segs - 1
    mfcc_tot = []
    audio_tot = []
    data_tot = []
    sf = 0
    
    processor = Wav2Vec2Processor.from_pretrained("/home/heqing001/Coding/SER_0915/features_extraction/pretrained_model/wav2vec2-base-960h")
    
    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        if end_wav > time_wav:
            end_wav = time_wav
            start_wav = max(0, end_wav - segment_size_wav)
        """
        if end-start < 100:
            num_segs -= 1
            print('truncated')
            break
        """
        # Do padding
        mfcc_pad = np.pad(
                mfcc[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
        
        audio_pad = np.pad(input_values[start_wav:end_wav], ((segment_size_wav - (end_wav - start_wav)), (0)), mode="constant")
  
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
                #data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant",
                #constant_values=((-80,-80),(-80,-80)))
            data_pad.append(data_ch)

        
        #audio_wav = processor(audio_wav.cpu(), sampling_rate=16000, return_tensors="pt").input_values# [1, batch, 48000] 
        #audio_wav = audio_wav.permute(1, 2, 0) # [batch, 48000, 1] 
        #audio_wav = audio_wav.reshape(audio_wav.shape[0],-1) # [batch, 48000] 
        
        
        data_pad = np.array(data_pad)
        
        # Stack
        mfcc_tot.append(mfcc_pad)
        data_tot.append(data_pad)

        audio_pad_np = np.array(audio_pad)
        audio_pad_pt = processor(audio_pad_np, sampling_rate=16000, return_tensors="pt").input_values
        audio_pad_pt = audio_pad_pt.view(-1)
        audio_pad_pt_np = audio_pad_pt.cpu().detach().numpy()
        audio_tot.append(audio_pad_pt_np)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)
        start_wav = end_wav
        end_wav = min(time_wav, end_wav + segment_size_wav)      
    
    mfcc_tot = np.stack(mfcc_tot)
    data_tot = np.stack(data_tot)
    audio_tot = np.stack(audio_tot)
    utt_label = emotion
    segment_labels = [emotion] * num_segs
    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, segment_labels, utt_label, mfcc_tot, audio_tot)

#Feature extraction function map
GET_FEATURES = {'logspec': extract_logspec,
                'logmelspec': extract_logmelspec,
                'logmeldeltaspec': extract_logmeldeltaspec,
                'lognrevspec': extract_lognrevspec,
                'logdeltaspec': extract_logdeltaspec
                }

if __name__ == '__main__':
    #test
    sig,sr = librosa.load('noise_wav/presto.wav', sr=None)

    params={'window': 'hamming',
            'win_length': 40,
            'hop_length': 10,
            'ndft':800,
            'nfreq':200}
    logdeltaspec = extract_logdeltaspec(sig[5000:37000], sr, params)
    data = segment_nd_features(logdeltaspec, None, 300)
    segment = data[1]
    segment = np.squeeze(segment)
    segment = np.squeeze(segment)
    print(segment.shape)
    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.specshow(segment[0])
    plt.subplot(3,1,2)
    librosa.display.specshow(segment[1])
    plt.subplot(3,1,3)
    librosa.display.specshow(segment[2])
    plt.show()

