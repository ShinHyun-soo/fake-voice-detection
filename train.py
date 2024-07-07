import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import pytorch_lightning as pl

from util import *

# Constants
DATA_DIR = './model'  
PREPROC_DIR = './preproc'
SUBMISSION_DIR = './submission'
MODEL_DIR = './model'
SAMPLING_RATE = 16000
SEED = 42
N_FOLD = 5
BATCH_SIZE = 4
NUM_LABELS = 2
#AUDIO_MODEL_NAME = 'abhishtagatya/hubert-base-960h-itw-deepfake'
#AUDIO_MODEL_NAME = 'abhishtagatya/hubert-base-960h-asv19-deepfake'
AUDIO_MODEL_NAME = 'facebook/hubert-base-ls960'

# Main script
if __name__ == '__main__':
    seed_everything(SEED)

    # 사운드 특징 추출
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_feature_extractor.return_attention_mask = True

    # 데이터 로드
    train_df = pd.read_csv('./train.csv')
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_DIR, x))

    # 싱글 라벨을 멀티 라벨로 변환
    train_df['label'] = train_df['label'].apply(
        lambda x: [1, 0] if x == 0 else (
            [0, 1] if x == 1 else (
                [1, 1] if x == 2 else (
                    [0, 0] if x == 3 else x
                )
            )
        )
    )

    train_audios, valid_indices = getAudios(train_df)
    train_df = train_df.iloc[valid_indices].reset_index(drop=True)
    train_labels = np.array(train_df['label'].tolist())

    # K 폴드
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_labels, train_labels.argmax(axis=1))):
        train_fold_audios = [train_audios[train_index] for train_index in train_indices]
        val_fold_audios = [train_audios[val_index] for val_index in val_indices]

        train_fold_labels = train_labels[train_indices]
        val_fold_labels = train_labels[val_indices]
        train_fold_ds = MyDataset(train_fold_audios, audio_feature_extractor, train_fold_labels)
        val_fold_ds = MyDataset(val_fold_audios, audio_feature_extractor, val_fold_labels)
        train_fold_dl = DataLoader(train_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        val_fold_dl = DataLoader(val_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        checkpoint_acc_callback = ModelCheckpoint(
            monitor='val_cs',  # Change to monitor combined_score
            dirpath=MODEL_DIR,
            filename=f'fold_{fold_idx}' + '_{epoch:02d}-{val_cs:.4f}',
            save_top_k=1,
            mode='min'  # Change to 'min' because lower combined_score is better
        )

        my_lit_model = MyLitModel(
            audio_model_name=AUDIO_MODEL_NAME,
            num_labels=NUM_LABELS,
            n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=0.8
        )

        trainer = pl.Trainer(
            accelerator='cuda',
            max_epochs=10,
            precision='16-mixed',
            callbacks=[checkpoint_acc_callback],
            accumulate_grad_batches=2 # batch_size * accumulate_grad_batches = 가 실질적인 배치 사이즈임.
        )

        trainer.fit(my_lit_model, train_fold_dl, val_fold_dl)

        del my_lit_model
