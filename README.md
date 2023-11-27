
# CA-MSER
Code for [Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information](https://arxiv.org/abs/2203.15326) (ICASSP 2022)

## NEW Update
The code for data processing is available online now. It can be downloaded and used as a reference.
<br>
If you think our paper and code are useful for your research work. Please give us a star or cite our original paper. This will give us the motivation to continue to share our code. 

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
 1. Download the pretrained Wav2vec2.0 model from https://huggingface.co/facebook/wav2vec2-base-960h
 2. Download the processed data. (It is a little big, later we will delete it from Google Drive)
 [Google Drive](https://drive.google.com/file/d/1Nnxh3y7hkkmsh3Y5Dg4q1qerRZWcePH8/view?usp=sharing); 
 [Baidu YunPan](https://pan.baidu.com/s/1MmmTrJ6nwQvlUiEWlDQhSw?pwd=q9gd)

 4. Install related libraries. pip install requirements.txt
 5. Run. python crossval_SER.py

### citation
If you use our code or find our CA-MSER useful in your research, please consider citing:

    @inproceedings{zou2022speech,
        title={Speech Emotion Recognition with Co-Attention Based Multi-Level Acoustic Information},
        author={Zou, Heqing and Si, Yuke and Chen, Chen and Rajan, Deepu and Chng, Eng Siong},
        booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={7367--7371},
        year={2022},
        organization={IEEE}
    }


