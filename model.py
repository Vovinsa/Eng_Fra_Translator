import torch

ENCODER_WEIGHTS_PATH = "modelWeights/encoder_w.pth"
DECODER_WEIGHTS_PATH = "modelWeights/decoder_w.pth"


class Encoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        self.embeddings = torch.nn.Embedding(self.input_size, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, inp, hidden):
        embedded = self.embeddings(inp).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.randn(1, 1, self.hidden_size, device=self.device)


class DecoderAttention(torch.nn.Module):
    def __init__(self, output_size: int, hidden_size: int, dropout_p=0.1, max_length=10):
        super().__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embeddings = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attention = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)

        self.rnn = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.relu = torch.nn.ReLU()

    def forward(self, inp, hidden, encoder_outputs):
        embedded = self.embeddings(inp).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = self.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1))
        )
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = self.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.log_softmax(self.out(output[0]))
        return output, hidden, attention_weights


def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(input_size=2803, hidden_size=256, device=device)
    decoder = DecoderAttention(output_size=4345, hidden_size=256)

    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_PATH, map_location=torch.device(device)))
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS_PATH, map_location=torch.device(device)))
    return encoder, decoder


# TODO
def predict(encoder: Encoder, decoder: DecoderAttention):
    pass
