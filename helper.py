import os
import tempfile
import torch
import numpy as np
from torch import nn
from azure.storage.blob import BlobServiceClient

# Model params
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 10


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def get_model_from_az_storage():
    model_path = 'checkpoint.pth.tar'

    # Get environment variable for Az Storage connection string to reference model
    if 'connect_str' in os.environ:
        connect_str = os.environ['connect_str']
    else:
        raise Exception('msg', 'connection string not found')

    # Get the model from Az Storage
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(
        container='gru-stock-container', blob='checkpoint.pth.tar')

    with open(os.path.join(tempfile.gettempdir(), model_path), "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

    checkpoint = torch.load(os.path.join(tempfile.gettempdir(),
                                         model_path), map_location=torch.device('cpu'))
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return checkpoint


def split_data(stock_val, lookback):
    data_raw = stock_val.values  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]


def preprocess_data(df, scaler):
    df_msft = df[['Close']]
    df_msft = df_msft.fillna(method='ffill')
    df_msft['Close'] = scaler.fit_transform(
        df_msft['Close'].values.reshape(-1, 1))
    return df_msft
