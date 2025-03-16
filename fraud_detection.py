import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

mlflow.set_experiment("Fraud_Detection_Experimet")

with mlflow.start_run():
    mlflow.log_param("learning rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 20)
    dtype_dict = {f'v{i}': 'float32' for i in range(1,29)}
    dtype_dict.update({'Time':'float32', 'Amount': 'float32', 'Class':'int8'})

    credit_card_df = pd.read_csv('creditcard.csv', dtype=dtype_dict)

    scalar = StandardScaler()
    credit_card_df['Amount'] = scalar.fit_transform(credit_card_df[['Amount']])

    X = credit_card_df.drop(columns=['Class'])
    y = credit_card_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    Y_train_np = y_train.values.astype(np.float32)
    Y_test_np = y_test.values.astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np).unsqueeze(-1))
    test_dataset = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np).unsqueeze(-1))

    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class FraudDetection(nn.Module):
        def __init__(self, input_dim):
            super(FraudDetection, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32,1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    input_dim = X_train_np.shape[1]
    model = FraudDetection(input_dim)

    classes = np.unique(Y_train_np)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=Y_train_np.flatten())
    criterion = nn.BCELoss(weight=torch.tensor([weights[1]], dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for batch_X, batch_Y in (train_dataloader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    model.eval()
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch_X, batch_Y in test_dataloader:
            outputs = model(batch_X)
            all_pred.extend(outputs.cpu().numpy())
            all_target.extend(batch_Y.cpu().numpy())

    scripted_model = torch.jit.script(model)
    scripted_model.save("model_scripted.pt")

    y_pred = (np.array(all_pred) > 0.5).astype(int)
    y_true = np.array(all_target).astype(int)

    print("Classification Report")
    print(classification_report(y_true, y_pred))
    roc_auc = roc_auc_score(y_true, np.array(all_pred))
    print(f"ROC AUC Score: {roc_auc:.4f}")
    mlflow.log_metric("roc_auc", roc_auc)