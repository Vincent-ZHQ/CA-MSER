
# CA-MSER
Code for Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information



## 1. File system
\- feature_extraction
<br> &ensp; -- pretrained_model
<br> &ensp; -- related python files
<br>
\- models
<br> &ensp;  -- transformers_encoder
<br> &ensp;  -- related python files
<br>
\- results
<br> &ensp;  -- t-SNE
<br>
\- crossval_SER.py
<br>
\- train_ser.py
<br>
\- data_utils.py
<br>
\- requirements.txt

## 2. Environmet
- PyTorch version:  1.8.0
- CUDA version:  11.1
- cudnn version:  8005
- GPU:  Tesla V100-SXM2-32GB

## 3. How to use
 1. Downlioad pretrained Wav2vec2.0 model from https://huggingface.co/facebook/wav2vec2-base-960h
 2. Downlioad the processed data. (It is a little big, later we will upload a smaller one for testing.)
 3. Install related libries. pip install requirements.txt
 4. Run. python crossval_SER.py