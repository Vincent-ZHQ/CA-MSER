# -*- coding: utf-8 -*-
import argparse
import array
import math
import numpy as np
import random
import wave
import librosa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_file', type=str, required=True)
    parser.add_argument('--noise_file', type=str, required=True)
    parser.add_argument('--output_mixed_file', type=str, default='', required=True)
    parser.add_argument('--output_clean_file', type=str, default='')
    parser.add_argument('--output_noise_file', type=str, default='')
    parser.add_argument('--snr', type=float, default='', required=True)
    args = parser.parse_args()
    return args

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

#if __name__ == '__main__':
def create_mixed_audio_file(clean_file, noise_file, output_mixed_file, snr, preemphasis=False):
    
    clean_wav,sr = librosa.load(clean_file, sr=None)
    #apply pre-emphasis on clean speech signal
    if preemphasis == True:
        clean_wav = librosa.effects.preemphasis(clean_wav, zi = [0.0])
    
    noise_wav,sr = librosa.load(noise_file, sr=None)
    
    # convert to 16-bit value range (32768.0 > amp >= -32768.0)
    clean_amp = np.round(clean_wav.astype(np.float64) * 32767.0, 0)
    noise_amp = np.round(noise_wav.astype(np.float64) * 32767.0, 0)
    
    clean_rms = cal_rms(clean_amp)

    start = random.randint(0, len(noise_amp)-len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 
    mixed_amp = (clean_amp + adjusted_noise_amp)

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
        clean_amp = clean_amp * (reduction_rate)

    #convert back to [-1.0, 1.0] range
    mixed_amp = (mixed_amp / 32767.0).astype(np.float32)
    librosa.output.write_wav(output_mixed_file, mixed_amp, sr)

if __name__ == '__main__':
    NOISE_FILES={'cafetaria.wav': 0,
             'car.wav': 0,
             'kitchen.wav': 0,
             'park.wav': 6,
             'traffic.wav': 6,
             'metro.wav': 0,
             'presto.wav': 10,
             'station.wav': 6,
             'river.wav':0,
             'square.wav':0}
    clean_wav = 'clean_wav/Ses01F_impro01_F011.wav'

    for noise in NOISE_FILES.keys():
        noise_wav = 'noise_wav/'+noise
        create_mixed_audio_file(clean_wav, noise_wav,
                                'clean_wav/speech_plus_'+noise, NOISE_FILES[noise],
                                preemphasis=False)

        #pre-emphasis on mixed signal

        mixed,sr = librosa.load('clean_wav/speech_plus_'+noise, sr=None)
        x_noise = librosa.effects.preemphasis(mixed, zi = [0.0])
        librosa.output.write_wav('clean_wav/speech_plus_preemp_'+noise,x_noise, sr )