import torch
import pandas as pd
import os
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
from skimage import io
from sklearn.metrics import f1_score, recall_score, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoadSignDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label

    def __len__(self):
        return len(self.annotations)


def check_accuracy(loader, model, type):
    num_correct = 0
    num_samples = 0
    model.eval()

    print(f"Checking {type} accuracy...")

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')

        f1 = f1_score(y_true=y.cpu(),
                      y_pred=predictions.cpu(),
                      average='macro')
        recall = recall_score(y_true=y.cpu(),
                              y_pred=predictions.cpu(),
                              average='macro')
        precision = precision_score(y_true=y.cpu(),
                                    y_pred=predictions.cpu(),
                                    average='macro')

        print(f'Got macro f1 {f1} with recall {recall} and precision {precision}')

        data = [type, float(num_correct) / float(num_samples) * 100, f1, recall, precision]

        with open('data/result/result.csv', 'a', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(data)

    model.train()


def save_checkpoint(state, filename='checkpoint/checkpoint.pth.tar'):
    print("Saving check point...")
    torch.save(state, filename)