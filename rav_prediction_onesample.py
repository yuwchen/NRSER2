import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
import torch.nn as nn
from pydub import AudioSegment

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
    
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())
    # this model is VERY SLOW, so best to pass in small sections that contain 
    # emotional words from the transcript. like 10s or less.
    # how to make sub-chunk  -- this was necessary even with very short audio files 
    # test = torch.tensor(input.input_values.float()[:, :100000])
    print(sound_array.shape)

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
    return interp

result = predict_emotion('./data/RAVDESS_Audio_Speech_Actors_01-24_renamed/Actor-19_03-01-05-01-02-02-19.wav')

print(result)
print(max(result, key=result.get))
