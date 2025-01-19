import os
import gc
import math
import argparse
import torch
import pysepm
import torch.nn as nn
import fairseq
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from utils import *
from snr_model_v3 import SNRLevelDetection

from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment

gc.collect()
torch.cuda.empty_cache()


def get_filepaths(directory):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 


model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

#"""
model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)


torch_state_dict = torch.load('./wav2vec2-lg-xlsr-en-speech-emotion-recognition/pytorch_model.bin', map_location=torch.device('cpu'))

model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']

model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']
#"""

def predict_emotion(audio_file):

    if  isinstance(audio_file, str):
        sound = AudioSegment.from_file(audio_file)
        sound = sound.set_frame_rate(16000)
        sound_array = np.array(sound.get_array_of_samples())
    else:
        sound_array = audio_file

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    result = model.forward(input.input_values.float())
    # making sense of the result 
    """
    id2label = {
        "0": "angry",
        "1": "calm",
        "2": "disgust",
        "3": "fearful",
        "4": "happy",
        "5": "neutral",
        "6": "sad",
        "7": "surprised"
    }
    """
    id2label = {
        "0": "A",
        "1": "C",
        "2": "D",
        "3": "F",
        "4": "H",
        "5": "N",
        "6": "S",
        "7": "U"
    }

    interp = dict(zip(id2label.values(), list(round(float(i),4) for i in result[0][0])))
    pred_emo = max(interp, key=interp.get)
    return pred_emo, result[0][0].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data', help='Path of your DATA/ directory')
    parser.add_argument('--datalist', type=str, default='./txtfile/ravdess-clean-noisy-ori_all.txt', help='Path of your DATA/ list')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')

    args = parser.parse_args()
    
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    datalist = args.datalist
    
    N_FFT = 400
    SSL_OUT_DIM = 512
    HOP=100

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))



    noise_model = SNRLevelDetection(N_FFT, HOP).to(device)
    noise_model.eval()
    noise_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'NOISE'+os.sep+'best')))

    print('Loading data')

    validset = open(datalist,'r').read().splitlines()
    outfile = my_checkpoint_dir.split("/")[-1]+datalist.split('/')[-1]


    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    prediction = open(os.path.join(output_dir, outfile), 'w')
    
    clean_dir = './data/RAVDESS_Audio_Speech_Actors_01-24_renamed/'

    print('Starting prediction')
    for filepath in tqdm(validset):
        filepath = filepath.split(';')[0]
        with torch.no_grad():
            filename = filepath.split("/")[-1]

            wav, sr = torchaudio.load(os.path.join(datadir, filepath))
            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
                sr = 16000

            if '_snr' in filename:
                wavname_gt = filename.split('_snr')[0]+'.wav'
                wav_gt, sr = torchaudio.load(os.path.join(clean_dir,  wavname_gt))
                wav_gt = torch.reshape(wav_gt,(-1, )).numpy()
            else:
                wav_gt, _ = torchaudio.load(os.path.join(datadir, filepath))
                wav_gt = torch.reshape(wav_gt,(-1, )).numpy()            

            wav = wav.to(device)
            wavdir_en = filepath.split(os.sep)[0]+'_en'
            wavpath_en = os.path.join(datadir, wavdir_en, os.path.basename(filepath))
            en_wav = torchaudio.load(wavpath_en)[0]            
            en_wav = en_wav.to(device)
            S = noise_model(wav, en_wav)
            snr_level_score = torch.clamp(S, min=0., max=1.)
            emo_input = (en_wav*(1-snr_level_score) + wav*snr_level_score)/2
            
            emo_input = torch.reshape(emo_input, (-1,))
            emo_input = emo_input.cpu().detach().numpy()
            try:
                stoi = pysepm.stoi(wav_gt, emo_input, 16000)
                pesq = pysepm.pesq(wav_gt, emo_input, 16000)[1]
            except:
                stoi=''
                pesq=''
           
            pred_c, score_C = predict_emotion(emo_input)
            snr_level_score = snr_level_score.cpu().detach().numpy()
            torch.cuda.empty_cache()
            output = "{}; {}; A:{}; V:{}; D:{}; {}; S:{}; PESQ:{}; STOI:{}".format(filename, pred_c, '', '', '', str(list(score_C)), snr_level_score, pesq, stoi)
            prediction.write(output+'\n')


if __name__ == '__main__':
    main()
