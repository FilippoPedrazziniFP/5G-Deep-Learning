import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# importing the dataset
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]

dataset = pd.read_csv('./dataset_kdd/train_20.csv', delimiter=',',header=None, names=col_names, index_col=False)

# numerical features
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

target = dataset['labels'].copy()
target[target != 'normal'] = 'attack'

X = dataset.iloc[:, :-1]
X = pd.get_dummies(X)

X = X.astype(float).values

# y_to_binary
le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y = binary_target
print(y.dtype)

# da 2D a 1D array
y.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print('dataset shape {}'.format(Counter(y_train)))

# Get dimensions of input and output
dimof_input = X_train.shape[1]
dimof_output = np.max(y_train) + 1
print('dimof_input: ', dimof_input)
print('dimof_output: ', dimof_output)

print(X_train.shape)
print(y_train.shape)

mini_batch = 128
epochs = 10

X_train = torch.from_numpy(X_train)
X_train = X_train.float()
y_train = torch.LongTensor(y_train)
train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=mini_batch, shuffle=True)


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.bn1(out)
        out = self.dp1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.bn2(out)
        out = self.dp2(out)
        out = self.fc3(out)
        return out


model = FeedForwardNeuralNetwork(input_dim=118, hidden_dim=100, output_dim=2)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (data, target) in enumerate(train_loader):
        inputs = Variable(data)
        labels = Variable(target)
        # clean the gradient
        optimizer.zero_grad()
        # forward to get output
        outputs = model(inputs)
        # calculate loss
        loss = criterion(outputs, labels)
        # getting gradients wrt of parameters
        loss.backward()
        # updating parameters
        optimizer.step()
    print("epoch {}, loss {}".format(epoch, loss.data[0]))

X_test = torch.from_numpy(X_test)
X_test = X_test.float()
inputs_test = Variable(X_test)

outputs_test = model(inputs_test)
_, predicted = torch.max(outputs_test.data, 1)
predicted = predicted.numpy()
print(predicted)

def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df

print("Confusion Matrix")
print(custom_confusion_matrix(y_test, predicted, ['Attack', 'Normal']))
print("Classification Report")
print(classification_report(y_test, predicted))

