import torch

from Langs import langs


ENCODER_WEIGHTS_PATH = "modelWeights/encoder_w.pth"
DECODER_WEIGHTS_PATH = "modelWeights/decoder_w.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_SIZE = 2803
OUTPUT_SIZE = 4345
HIDDEN_SIZE = 256

SOS_token = 0
EOS_token = 1


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

        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
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
        output, hidden = self.gru(output, hidden)
        output = self.log_softmax(self.out(output[0]))
        return output, hidden, attention_weights


def create_model(max_length: int):
    encoder = Encoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, device=DEVICE)
    decoder = DecoderAttention(output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE, max_length=max_length)

    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_PATH, map_location=torch.device(DEVICE)))
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS_PATH, map_location=torch.device(DEVICE)))
    return encoder, decoder


def predict(encoder: Encoder, decoder: DecoderAttention, sentence: str, words_dict: dict, indexes_dict: dict, max_length: int):
    with torch.no_grad():
        input_tensor = langs.read_sentence(sentence, words_dict)
        input_size = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for idx in range(input_size):
            encoder_output, encoder_hidden = encoder(input_tensor[idx].unsqueeze(0), encoder_hidden)
            encoder_outputs[idx] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)

        decoder_hidden = encoder_hidden
        decoded_words = []

        for idx in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_value, top_index = decoder_output.topk(1)
            if top_index.item() == EOS_token:
                break
            else:
                decoded_words.append(langs.index2word(top_index.item(), indexes_dict))
            decoder_input = top_index.squeeze().detach()
        return " ".join(decoded_words)

