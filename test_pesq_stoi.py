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

    parser.add_argument('--datadir', type=str, default='./data', help='Path of your DATA/ directory')
    parser.add_argument('--datalist', type=str, default='./txtfile/msp1_11-test2-clean-noisy.txt', help='Path of your DATA/ list')

    args = parser.parse_args()
    
    datadir = args.datadir
    datalist = args.datalist
    
    N_FFT = 400

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))
    

    print('Loading data')
    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    validset = open(datalist,'r').read().splitlines()

    snr_level_list = [ 0, 0.25, 0.5, 0.75, 1]
    
    clean_dir = './data/Audios/'

    for snr_level_score in snr_level_list:

        outfile = 'S_'+str(snr_level_score)+'_'+datalist.split('/')[-1]
        prediction = open(os.path.join(output_dir, outfile), 'w')
        
        print('Starting prediction')
        for filepath in tqdm(validset):
            
            filepath = filepath.split(";")[0]
            filename = filepath.split("/")[-1]

            wav, sr = torchaudio.load(os.path.join(datadir, filepath))

            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
                sr = 16000

            wav = torch.reshape(wav,(-1, )).numpy()
            if '_snr' in filename:
                wavname_gt = filename.split('_snr')[0]+'.wav'
                wav_gt, sr = torchaudio.load(os.path.join(clean_dir, wavname_gt))
                wav_gt = torch.reshape(wav_gt,(-1, )).numpy()
            else:
                wav_gt, _ = torchaudio.load(os.path.join(datadir, filepath))
                wav_gt = torch.reshape(wav_gt,(-1, )).numpy()
                 
            wavdir_en = filepath.split(os.sep)[0]+'_en'
            wavpath_en = os.path.join(datadir, wavdir_en, os.path.basename(filepath))
            en_wav = torchaudio.load(wavpath_en)[0]          
            en_wav = torch.reshape(en_wav,(-1, ))    

            emo_input = (en_wav*(1-snr_level_score) + wav*snr_level_score)/2
            emo_input = emo_input.numpy()
            emo_input = np.reshape(emo_input, (-1, ))
            try:
                stoi = pysepm.stoi(wav_gt, emo_input, 16000)
                pesq = pysepm.pesq(wav_gt, emo_input, 16000)[1]
            
                output = "{}; S_raw:{}; S:{}; stoi:{}, pesq:{}".format(filename, '', snr_level_score, stoi, pesq)
                prediction.write(output+'\n')
            except Exception as e:
                
                output = "{}; S_raw:{}; S:{}; stoi:{}, pesq:{}".format(filename, '', snr_level_score, '', '')
                prediction.write(output+'\n')
                print(e)
 


if __name__ == '__main__':
    main()
