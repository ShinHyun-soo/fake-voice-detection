import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import pytorch_lightning as pl

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
from transformers import HubertForSequenceClassification, AutoFeatureExtractor, AutoConfig
from torch.optim import AdamW
import bitsandbytes as bnb


from util import *

# Main script
if __name__ == '__main__':
    seed_everything(SEED)

    # 사운드 특징 추출
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_feature_extractor.return_attention_mask = True

    # 데이터 로드
    test_df = pd.read_csv('./test.csv')
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_DIR, x))

    test_df['label'] = [[0, 0]] * len(test_df)

    # 테스트 셋 예측
    test_audios, _ = getAudios(test_df)
    test_ds = MyDataset(test_audios, audio_feature_extractor)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    pretrained_models = list(map(lambda x: os.path.join(MODEL_DIR, x), os.listdir(MODEL_DIR)))
    
    test_preds = []
    trainer = pl.Trainer(
        accelerator='cuda',
        precision='16',
    )
    
    for pretrained_model_path in pretrained_models:
        pretrained_model = MyLitModel.load_from_checkpoint(
            pretrained_model_path,
            audio_model_name=AUDIO_MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        test_pred = trainer.predict(pretrained_model, test_dl)
        test_pred = torch.cat(test_pred).detach().cpu().numpy()
        test_preds.append(test_pred)
      
        del pretrained_model
    
    # shape: (num_models, num_samples, num_classes)
    test_preds = np.array(test_preds)
    mean_preds = np.mean(test_preds, axis=0)
  
    # 0열 값을 fake, 1열 값을 real
    submission_df = pd.read_csv(os.path.join('sample_submission.csv'))
    submission_df['fake'] = mean_preds[:, 0]
    submission_df['real'] = mean_preds[:, 1]
    submission_df.to_csv(os.path.join(SUBMISSION_DIR, 'fold_5_ensemble.csv'), index=False)
