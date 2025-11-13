import pandas as pd
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from HyperParameter import HyperParameter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from MultiAnchor import MultiAnchorLoss, MultiAnchorMiner
import argparse


# Define a custom dataset for metric learning
class MetricDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


# Defines the metric learning and data stream handling
class MetricStream(object):
    def __init__(self, args):
        self.args = args
        if self.args.metric_learning:
            if self.args.retrain or not os.path.exists(f'model/{self.args.task}.bin'):
                self.metric_learning()

        init_stream = np.load(f'data/init_{self.args.task}.npy')
        self.init_data = init_stream[:, :-1]
        self.init_labels = init_stream[:, -1].astype(np.int8)

        metric_stream = np.load(f'data/{self.args.task}.npy')
        self.metric_stream = metric_stream[1000:, :-1]
        self.true_labels = metric_stream[1000:, -1].astype(np.int8)
        self.semi_labels = self.true_labels.copy()

        # Create semi-supervised labels by masking a portion of true labels
        np.random.seed(self.args.seed)
        index = np.random.choice(np.arange(len(self.true_labels)),
                                 size=int(self.args.unlabeled_ratio * len(self.true_labels)),
                                 replace=False)
        self.semi_labels[index] = -1
        self.classes = list(set(list(self.true_labels)))
        print('data stream labels: ', self.classes)

    def metric_learning(self):
        x_init, x_stream, y_init, y_stream = self.__preprocess()

        in_dim = x_init.shape[1]
        x_train, x_val, y_train, y_val = train_test_split(x_init, y_init, test_size=0.2)

        train_dataset = MetricDataset(x_init, y_init)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size)
        val_dataset = MetricDataset(x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.eval_batch_size)

        # Set model embedding dimensions based on input dimension
        if in_dim >= 256:
            dim_l1 = int(0.6 * in_dim)
            dim_l2 = int(0.3 * in_dim)
            dim_out = 64
        elif in_dim <= 32:
            dim_l1 = int(4 * in_dim)
            dim_l2 = int(2 * in_dim)
            dim_out = 8
        else:
            dim_l1 = 128
            dim_l2 = 64
            dim_out = 32

        cuda = False
        if self.args.metric_learning:
            if self.args.retrain or not os.path.exists(f'model/{self.args.task}.bin'):
                best_accuracy = 0
                best_model = None
                accuracies = []
                # Train multiple models and select the best one
                for _ in range(10):
                    model = self.args.model(in_dim, dim_l1, dim_l2, dim_out)
                    if self.args.device.startswith('cuda') and torch.cuda.is_available():
                        model.to(self.args.device)
                        cuda = True

                    miner = MultiAnchorMiner(num_hard_samples=1, k=7)
                    loss_func = MultiAnchorLoss(margin1=1.0, margin2=0.5, margin3=0.5, lambda_reg=0.01)

                    model = self.__train_model(train_dataloader, miner, loss_func, model, cuda)

                    model.eval()
                    with torch.no_grad():
                        X_train = torch.Tensor(x_train)
                        X_train = X_train.to(self.args.device) if cuda else X_train
                        embeddings = model(X_train).cpu().detach().numpy()
                        labels = y_train

                        X_val = torch.Tensor(x_val)
                        X_val = X_val.to(self.args.device) if cuda else X_val
                        embeddings_val = model(X_val).cpu().detach().numpy()
                        labels_val = y_val

                    # Use KNN classifier to evaluate embedding quality
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(embeddings, labels)
                    predicted = knn.predict(embeddings_val)
                    accuracy = accuracy_score(labels_val, predicted)
                    accuracies.append(accuracy)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                self.__save(best_model)
                print("Validation accuracies:", accuracies)
                # Convert init and stream data to new embedding space
                self.__convert_stream(best_model, x_init, y_init, cuda, init=True)
                self.__convert_stream(best_model, x_stream, y_stream, cuda, init=False)
            else:
                model = self.__load()

        self.evaluation(x_init, y_init, cuda, model)

    def __preprocess(self):
        # Load data from CSV and split into initial and stream parts
        data = pd.read_csv(self.args.data_dir, header=None, dtype='float32')
        print(data.shape)
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values
        Y = Y.astype(np.int8)

        self.labels = list(set(Y))
        train_datasize = self.args.train_datasize if self.args.train_datasize > 1 else self.args.train_datasize * len(Y)

        x_init, y_init = [], []
        for y_temp in self.labels:
            x = X[Y == y_temp]
            y = Y[Y == y_temp]
            x_init.extend(x[:int(train_datasize / len(self.labels))])
            y_init.extend(y[:int(train_datasize / len(self.labels))])
        x_init, y_init = np.stack(x_init), np.stack(y_init)
        x_stream, y_stream = X, Y

        init_index = np.arange(len(y_init))
        np.random.shuffle(init_index)
        x_init, y_init = x_init[init_index], y_init[init_index]
        return x_init, x_stream, y_init, y_stream

    def __train_model(self, train_dataloader, miner, loss_func, model, cuda):
        # Standard model training loop for metric learning
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.learning_rate)
        for epoch in tqdm(range(self.args.metric_epoch)):
            for data, labels in train_dataloader:
                optimizer.zero_grad()
                data = data.float()
                if cuda:
                    data = data.to(self.args.device)
                    labels = labels.to(self.args.device)
                embeddings = model(data)
                mining_results = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, mining_results)
                loss.backward()
                optimizer.step()
        return model

    def __convert_stream(self, model, data, labels, cuda, init):
        # Save the transformed data (embeddings) to .npy files
        model.eval()
        with torch.no_grad():
            X = torch.Tensor(data)
            X = X.to(self.args.device) if cuda else X
            X = model(X).cpu().detach().numpy()
            y = labels

            stream = np.hstack([X, labels.reshape([-1, 1])])

            if init:
                np.save(os.path.join('data', 'init_' + self.args.task + '.npy'), stream)
            else:
                np.save(os.path.join('data', self.args.task + '.npy'), stream)

    def evaluation(self, data, labels, cuda, model):
        # Generate t-SNE visualizations for embeddings and original data
        X = torch.Tensor(data)
        X = X.to(self.args.device) if cuda else X
        X = model(X).cpu().detach().numpy()
        y = labels

        X_std = StandardScaler().fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_std)

        X_tsne_data = np.vstack((X_tsne.T, y)).T
        df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
        plt.savefig(f'tsne/{self.args.task}_latent.png')

        X_original_std = StandardScaler().fit_transform(data)
        tsne_original = TSNE(n_components=2, random_state=42)
        X_original_tsne = tsne_original.fit_transform(X_original_std)

        X_original_tsne_data = np.vstack((X_original_tsne.T, y)).T
        df_original_tsne = pd.DataFrame(X_original_tsne_data, columns=['Dim1', 'Dim2', 'class'])
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_original_tsne, hue='class', x='Dim1', y='Dim2')
        plt.savefig(f'tsne/{self.args.task}_original.png')

    def __load(self):
        mode = torch.load(f'model/{self.args.task}.bin')
        return mode

    def __save(self, model):
        torch.save(model, f'model/{self.args.task}.bin')


if __name__ == '__main__':
    args = HyperParameter()
    ms = MetricStream(args)