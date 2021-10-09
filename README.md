# CA-MSER
Code for Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information



## 1. File system
- feature_extraction
  -- pretrained_model
  -- related python files
- models
  -- transformers_encoder
  -- related python files
- results
  -- t-SNE
- crossval_SER.py
- train_ser.py
- data_utils.py
- requirements.txt

## 2. Environmet
pytorch version:  1.8.0
cuda version:  11.1
cudnn version:  8005
gpu name:  Tesla V100-SXM2-32GB

## 3. How to use
 1. Downlioad pretrained Wav2vec2.0 model from https://huggingface.co/facebook/wav2vec2-base-960h
 2. Downlioad the processed data.
 3. Install requirements.txt
 4. python crossval_SER.py
 
