import os
import argparse
import fairseq
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from dataloader import MyDataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

random.seed(1984)


class EmotionPredictor(nn.Module):
    
    def __init__(self, ssl_model, ssl_out_dim):
        super(EmotionPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer_A = nn.Linear(self.ssl_features, 1) #arousal
        self.output_layer_V = nn.Linear(self.ssl_features, 1) #valence
        self.output_layer_D = nn.Linear(self.ssl_features, 1) #dominance
        self.output_layer_C = nn.Linear(self.ssl_features, 10) #categorical

        self.predicted_ids = torch.tensor([[1, 1]]) * self.ssl_model.config.decoder_start_token_id
        self.predicted_ids = self.predicted_ids.to('cuda')

    def forward(self, wav):

        features = self.ssl_model(wav, decoder_input_ids=self.predicted_ids)
        x = features.encoder_last_hidden_state
        x = torch.mean(x, 1)
        output_A = self.output_layer_A(x)
        output_V = self.output_layer_V(x)
        output_D = self.output_layer_D(x)
        output_C = self.output_layer_C(x)
        
        return output_A.squeeze(1), output_V.squeeze(1), output_D.squeeze(1), output_C


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./data',  type=str, help='Path to root data directory')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path to training txt directory')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to the checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='emotion_model_whisper_noisy_b6_r1', help='Output directory for your trained checkpoints')

    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    txtfiledir = args.txtfiledir
    my_checkpoint_dir = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir,'EMO'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, '')
    trainlist = os.path.join(txtfiledir, 'msp1_11-train-clean-noisy.txt')
    validlist = os.path.join(txtfiledir, 'msp1_11-dev-clean-noisy.txt')

    SSL_OUT_DIM = 512

    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    whisper_model.config.forced_decoder_ids = None
    ssl_model = whisper_model

    
    trainset = MyDataset(wavdir, trainlist,'whisper')
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)
    validset = MyDataset(wavdir, validlist,'whisper')
    validloader = DataLoader(validset, batch_size=6, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    emo_net = EmotionPredictor(ssl_model, SSL_OUT_DIM)
    emo_net = emo_net.to(device)

    if my_checkpoint_dir != None:  ## do (further) finetuning
        emo_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'EMO','best')))

    criterion_cross = nn.CrossEntropyLoss() #for categorical emotion prediction
    optimizer_emo = optim.SGD( emo_net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    orig_patience=2
    patience=orig_patience

    for epoch in range(1,1001):
        STEPS=0
        emo_net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, _, scores_A, scores_V, scores_D, scores_C, _ = data
            inputs = inputs.to(device)
            scores_A = scores_A.to(device)
            scores_V = scores_V.to(device)
            scores_D = scores_D.to(device)
            scores_C = scores_C.to(device)

            emo_input = inputs.squeeze(1)  
            optimizer_emo.zero_grad()
            output_A, output_V, output_D, output_C = emo_net(emo_input)
            loss_A = ccc_loss(output_A, scores_A)
            loss_V = ccc_loss(output_V, scores_V)
            loss_D = ccc_loss(output_D, scores_D)
            loss_C = criterion_cross(output_C, scores_C)
            
            loss = loss_C +  (loss_A + loss_V + loss_D )/3
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer_emo.step()
            STEPS += 1
            running_loss += loss.item()

        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))


        ## validation
        VALSTEPS=0
        epoch_val_loss = 0.0
        emo_net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 

        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, _, scores_A, scores_V, scores_D, scores_C, _ = data
            inputs = inputs.to(device)
            scores_A = scores_A.to(device)
            scores_V = scores_V.to(device)
            scores_D = scores_D.to(device)
            scores_C = scores_C.to(device)
            emo_input = inputs.squeeze(1)

            with torch.no_grad(): 
                output_A, output_V, output_D, output_C = emo_net(emo_input)
                loss_A = ccc_loss(output_A, scores_A)
                loss_V = ccc_loss(output_V, scores_V)
                loss_D = ccc_loss(output_D, scores_D)
                loss_C = criterion_cross(output_C, scores_C)
                
                loss = loss_C + (loss_A + loss_V + loss_D)/3
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(emo_net.state_dict(), os.path.join(ckptdir,'EMO','best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training of Emotion Model')

if __name__ == '__main__':
    main()
