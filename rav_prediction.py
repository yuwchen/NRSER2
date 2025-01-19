import os
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
import torch.nn as nn
from pydub import AudioSegment
import torchaudio
from tqdm import tqdm

# https://github.com/ehcalabres/EMOVoice
# the preprocessor was derived from https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english
# processor1 = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# ^^^ no preload model available for this model (above), but the `feature_extractor` works in place
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

    # this model is VERY SLOW, so best to pass in small sections that contain 
    # emotional words from the transcript. like 10s or less.
    # how to make sub-chunk  -- this was necessary even with very short audio files 
    # test = torch.tensor(input.input_values.float()[:, :100000])

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


datadir = './data'
clean_dir = './RAVDESS_Audio_Speech_Actors_01-24_renamed'
file_list = open('./txtfile/ravdess-clean-noisy-ori_all.txt','r').read().splitlines()
snr_level_list = [ 0, 0.25, 0.5, 0.75, 1]

for snr_level_score in snr_level_list:

    outfile = 'wav2vec2-lg-xlsr-en-speech-emotion-recognition_snr'+str(snr_level_score)+'_ravdess'
    prediction = open(os.path.join('Results', outfile), 'w')

    for line in tqdm(file_list):

        filepath = line.split(';')[0]
        filename = filepath.split("/")[-1]
        wavpath = os.path.join(datadir, filepath)
        wav, sr = torchaudio.load(os.path.join(datadir, filepath))

        if sr!=16000:
            transform = torchaudio.transforms.Resample(sr, 16000)
            wav = transform(wav)
            sr = 16000

        wav = torch.reshape(wav,(-1, )).numpy()

        wavdir_en = filepath.split(os.sep)[0]+'_en'
        wavpath_en = os.path.join(datadir, wavdir_en, os.path.basename(filepath))
        en_wav = torchaudio.load(wavpath_en)[0]
        en_wav = torch.reshape(en_wav,(-1, ))

        emo_input = (en_wav*(1-snr_level_score) + wav*snr_level_score)/2
        emo_input = emo_input.numpy()
        emo_input = np.reshape(emo_input, (-1, ))

        pred_c, score_C = predict_emotion(emo_input)
        output = "{}; {}; A:{}; V:{}; D:{}; {}; S:{}".format(filename, pred_c, '','', '', str(list(score_C)), snr_level_score)
        prediction.write(output+'\n')

