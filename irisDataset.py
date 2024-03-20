import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target 

#map target names to target labels
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Pairplot to visualize pairwise relationships in the dataset
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'D'])
# plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal width (cm)', data=iris_df)
plt.title('Distribution of Petal Width for each Iris Species')
# plt.show()


X, y = iris.data, iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Number of samples:", len(X)) 
print("Number of features:", X.shape[1])  # X has 150 rows 4 cols, this gets cols
# print(X[:5])
# print(y[:5])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc_3(self.relu(self.fc_2(self.relu(self.fc_1(x)))))

input_size = X_train.shape[1]
hidden_size = 10
num_classes=3

model = NeuralNet(input_size, hidden_size, num_classes)

# Step 3: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if(epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Percentage loss: {(loss.item() * 100):.2f}%')

model.eval()
with torch.no_grad():
    outputs = model(X_val)
    values, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_val).sum().item() / len(y_val)
    print(f'Accuracy on validation set: {accuracy * 100:.2f}%')
    print(f'Loss on validation set: {loss:.4f}')