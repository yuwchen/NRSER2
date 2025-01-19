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
from SEmodels import generator
from enhancement_model import SpeechEnhancement
from snr_model_v3 import SNRLevelDetection

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
    parser.add_argument('--fairseq_base_model', type=str, default='../pronunciation/fairseq/hubert_base_ls960.pt', help='Path to pretrained fairseq base model.')
    parser.add_argument('--cmgan_base_model', type=str, default='./CMGAN/best_ckpt/ckpt', help='Path to pretrained CMGAN model.')
    parser.add_argument('--datadir', type=str, default='./data', help='Path of your DATA/ directory')
    parser.add_argument('--datalist', type=str, default='./txtfile/ravdess-clean-noisy-s2.txt', help='Path of your DATA/ list')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')
    parser.add_argument('--savewav', type=bool, help='Whether to save the enhanced wav')

    args = parser.parse_args()
    
    ssl_path = args.fairseq_base_model
    se_path = args.cmgan_base_model
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    savewav = args.savewav
    datalist = args.datalist
    
    N_FFT = 400
    SSL_OUT_DIM = 768
    HOP=100

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))




    noise_model = SNRLevelDetection(N_FFT, HOP).to(device)
    noise_model.eval()
    noise_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'NOISE'+os.sep+'best')))

    print('Loading data')

    validset = open(datalist,'r').read().splitlines()
    outfile = 'S_'+my_checkpoint_dir.split("/")[-1]+datalist.split('/')[-1]


    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if savewav:
        wav_output_dir = os.path.join('Enhanced_wav',my_checkpoint_dir.split(os.sep)[-1])
        if not os.path.exists(wav_output_dir):
            os.makedirs(wav_output_dir)

    prediction = open(os.path.join(output_dir, outfile), 'w')
    
    clean_dir = './data/RAVDESS_Audio_Speech_Actors_01-24_renamed/'

    print('Starting prediction')
    for filepath in tqdm(validset):
        
        filepath = filepath.split(";")[0]
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
                wav_gt,_ = torchaudio.load(os.path.join(datadir, filepath))
                wav_gt = torch.reshape(wav_gt,(-1, )).numpy() 

            wav = wav.to(device)
            
            wavdir_en = filepath.split(os.sep)[0]+'_en'
            wavpath_en = os.path.join(datadir, wavdir_en, os.path.basename(filepath))
            en_wav = torchaudio.load(wavpath_en)[0]          
            en_wav = en_wav.to(device)     

            S = noise_model(wav, en_wav)
            snr_level_score = torch.clamp(S, min=0., max=1.)
            emo_input = (en_wav*(1-snr_level_score) + wav*snr_level_score)/2
            
            emo_input = emo_input.cpu().detach().numpy()
            emo_input = np.reshape(emo_input, (-1, ))

            stoi = pysepm.stoi(wav_gt, emo_input, 16000)
            pesq = pysepm.pesq(wav_gt, emo_input, 16000)[1]

            snr_level_score = snr_level_score.cpu().detach().numpy()
            S = S.cpu().detach().numpy()

            torch.cuda.empty_cache()
            output = "{}; S_raw:{}; S:{}; stoi:{}, pesq:{}".format(filename, S, snr_level_score, stoi, pesq)
            prediction.write(output+'\n')
 


if __name__ == '__main__':
    main()
