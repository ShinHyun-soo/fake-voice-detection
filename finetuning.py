import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch, random, os
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from mixup import mixup_data, mixup_criterion
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor, PretrainedConfig, HubertForSequenceClassification,AutoProcessor, Wav2Vec2ForCTC
import librosa
import IPython.display as ipd
from tqdm import tqdm
import numpy as np
import pandas as pd


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything(42)


df = pd.read_csv('data/train.csv', index_col=None)
df['path'] = 'data' + df['path'].str[1:]

test_df = pd.read_csv('data/test.csv', index_col=None)
test_df['path'] = 'data' + test_df['path'].str[1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = 'facebook/hubert-large-ll60k'
config = AutoConfig.from_pretrained(model_name_or_path, num_labels = 6)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    speech = speech_array.squeeze().numpy()
    return speech

class EModel(nn.Module):
    def __init__(self):
        super(EModel, self).__init__()
        self.backbone = HubertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    def forward(self, x):
        return self.backbone(x).logits

    model = EModel().to(device)

class EMDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        signal = speech_file_to_array_fn(path)
        label = self.df.loc[idx, 'label']
        return signal, label

k_split = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

def collate_fn(batch):
    signal = [i[0] for i in batch]
    label = [i[1] for i in batch]

    return signal, torch.tensor(label)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)


def metrics(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    f1s = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return f1s, acc


def trainer(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = 0
    step = 0
    for inputs, labels in train_loader:
        inputs = feature_extractor(inputs, sampling_rate=sampling_rate, padding=True, return_tensors='pt')[
            'input_values'].to(device)
        labels = labels.to(device)

        if step % 4 == 0:
            # if random.random() > 0.5:
            x_batch, y_batch_a, y_batch_b, lam = mixup_data(inputs, labels)
            # else:
            # x_batch, y_batch_a, y_batch_b, lam = cutmix_data(inputs, labels)

            outputs = model(x_batch)
            loss = mixup_criterion(loss_fn, outputs, y_batch_a.to(device), y_batch_b.to(device), lam)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().item()
        step += 1
    print(f'EPOCH : {epoch} | train_loss : {train_loss / len(train_loader):.4f}')


def validator(model, valid_loader, loss_fn, epoch, k, scheduler):
    model.eval()
    best_score = 0
    valid_loss = 0
    valid_labels = []
    valid_preds = []
    for inputs, labels in valid_loader:
        inputs = feature_extractor(inputs, sampling_rate=sampling_rate, padding=True, return_tensors='pt')[
            'input_values'].to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        valid_labels.extend(labels.cpu().tolist())
        valid_preds.extend(outputs.detach().cpu().argmax(1).tolist())
        valid_loss += loss.detach().cpu().item()
    f1s, acc = metrics(valid_labels, valid_preds)

    if acc > best_score:
        best_score = acc
        torch.save(model.state_dict(), f'{k}_best.pt')
    print(f'EPOCH : {epoch} | valid_loss : {valid_loss / len(valid_loader):.4f} | f1s : {f1s:.4f} | acc :{acc:.4f}')

    scheduler.step()

for k, (t_idx, v_idx) in enumerate(k_split.split(df, df['label'])):
    train_df, valid_df = df.loc[t_idx].reset_index(drop=True), df.loc[v_idx].reset_index(drop=True)

    train_dataset = EMDataset(train_df)
    valid_dataset = EMDataset(valid_df)

    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=8, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)
    for epoch in range(50):
        trainer(model, train_loader, loss_fn, optimizer, epoch)
        validator(model, valid_loader, loss_fn, epoch, k, scheduler)


class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        signal = speech_file_to_array_fn(path)
        return signal, -1

test_dataset = TestDataset(test_df)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=1, collate_fn=collate_fn)

sub = pd.read_csv('data/sample_submission.csv', index_col=None)

k_test_preds = []
for k in range(5):
    model = EModel().to(device)
    model.load_state_dict(torch.load(f'{k}_best.pt', map_location='cpu'))
    model.eval()

    test_preds = []
    for inputs, _ in tqdm(test_loader):
        inputs = feature_extractor(inputs, sampling_rate=sampling_rate, padding=True, return_tensors='pt')[
            'input_values'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        test_preds.extend(outputs.detach().cpu().tolist())
    k_test_preds.append(test_preds)
k_test_preds = torch.tensor(k_test_preds)

sub['label'] = torch.nn.functional.softmax(k_test_preds, 1).mean(0).argmax(1).tolist()
sub.to_csv('sub.csv', index=None)