import os
import gc
import math
import argparse
import torch
import torch.nn as nn
import fairseq
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from utils import *
from SEmodels import generator
from emotion_model_whisper_en import EmotionPredictor
from enhancement_model import SpeechEnhancement
from snr_model import SNRLevelDetection
from transformers import WhisperProcessor
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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

def en_one_track(noisy, sr, se_model, device):
    
    assert sr == 16000
    seg_length = 2*sr
    wav_len = noisy.size(-1)
    num_of_seg = math.ceil(wav_len / seg_length) 
    amount_to_pad = num_of_seg*seg_length - wav_len
    noisy = torch.nn.functional.pad(noisy, (0, amount_to_pad), 'constant', 0)
    enhanced_wav = torch.zeros(noisy.shape).to(device)

    noisy = noisy.cuda()

    for i in range (num_of_seg):
        wav_seg = noisy[:,i*seg_length:(i+1)*seg_length]
        en_seg = se_model(wav_seg)
        enhanced_wav[:,i*seg_length:(i+1)*seg_length] = en_seg
    
    enhanced_wav = enhanced_wav[:,:wav_len]
    return enhanced_wav

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmgan_base_model', type=str, default='./CMGAN/best_ckpt/ckpt', help='Path to pretrained CMGAN model.')
    parser.add_argument('--datadir', type=str, default='./data', help='Path of your DATA/ directory')
    parser.add_argument('--datalist', type=str, default='./txtfile/msp1_11-test2-clean-noisy.txt', help='Path of your DATA/ list')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')
    parser.add_argument('--savewav', type=bool, help='Whether to save the enhanced wav')

    args = parser.parse_args()
    
    se_path = args.cmgan_base_model
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    savewav = args.savewav
    datalist = args.datalist
    
    N_FFT = 400
    SSL_OUT_DIM = 512
    HOP=100

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")

    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    whisper_model.config.forced_decoder_ids = None
    ssl_model = whisper_model
    
    cmgan = generator.TSCNet(num_channel=64, num_features=N_FFT//2+1).cuda()
    cmgan.load_state_dict(torch.load(se_path))

    emo_model = EmotionPredictor(ssl_model, SSL_OUT_DIM).to(device)
    emo_model.eval()
    emo_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'EMO'+os.sep+'best')))


    print('Loading data')

    validset = open(datalist,'r').read().splitlines()
    outfile = my_checkpoint_dir.split("/")[-1]+datalist.split('/')[-1].replace('.txt','_en.txt')


    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if savewav:
        wav_output_dir = os.path.join('Enhanced_wav',my_checkpoint_dir.split(os.sep)[-1])
        if not os.path.exists(wav_output_dir):
            os.makedirs(wav_output_dir)

    prediction = open(os.path.join(output_dir, outfile), 'w')
    label_map = {0:'A',1:'S',2:'H',3:'U',4:'F',5:'D',6:'C',7:'N',8:'O',9:'X'}
    

    print('Starting prediction')
    for filepath in tqdm(validset):
        filepath = filepath.split(';')[0]
        with torch.no_grad():
            filename = filepath.split("/")[-1]

            
            wavdir_en = filepath.split(os.sep)[0]+'_en'
            wavpath_en = os.path.join(datadir, wavdir_en, os.path.basename(filepath))
            en_wav, sr = torchaudio.load(wavpath_en)           
            en_wav = torch.reshape(en_wav, (-1,))
            en_wav = processor(en_wav, sampling_rate=sr, return_tensors="pt").input_features
            en_wav = en_wav.to(device)              

            
            score_A, score_V, score_D, score_C = emo_model(en_wav)
            score_A = score_A.cpu().detach().numpy()[0]
            score_V = score_V.cpu().detach().numpy()[0]
            score_D = score_D.cpu().detach().numpy()[0]
            score_C = score_C.cpu().detach().numpy()[0]
            pred_c = label_map[np.argmax(score_C)]

            torch.cuda.empty_cache()
            output = "{}; {}; A:{}; V:{}; D:{}; {}; S:{}".format(filename, pred_c, score_A, score_V, score_D, str(list(score_C)), '')
            prediction.write(output+'\n')
 


if __name__ == '__main__':
    main()
