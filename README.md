
# CA-MSER
Code for [Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information](https://arxiv.org/abs/2203.15326) (ICASSP 2022)


## 1. File system
\- models
<br> &ensp;  -- transformers_encoder
<br> &ensp;  -- related python files
<br>
\- results
<br> &ensp;  -- t-SNE
<br>
\- extracted_features.pkl
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
 [Google Drive](https://drive.google.com/file/d/1B0RU9jANAKbUfPG4q8iaq67X6x8yOwu9/view?usp=sharing)
 [Baidu YunPan](https://pan.baidu.com/s/1MmmTrJ6nwQvlUiEWlDQhSw?pwd=q9gd)

 4. Install related libries. pip install requirements.txt
 5. Run. python crossval_SER.py

### citation
If you use our code or find our CA-MSER useful in your research, please consider citing:

    @article{zou2022speech,
      title={Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information},
      author={Zou, Heqing and Si, Yuke and Chen, Chen and Rajan, Deepu and Chng, Eng Siong},
      journal={arXiv preprint arXiv:2203.15326},
      year={2022}
    }


