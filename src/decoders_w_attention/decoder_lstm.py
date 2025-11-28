import torch
import torch.nn as nn
from final_project.src.encoders.encoder_cnn import EncoderCNN
from final_project.src.decoders_w_attention.attention import AttentionModule


class DecoderLSTM(nn.Module):
    """
    LSTM Decoder
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm_decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.init_hidden = nn.Linear(embed_dim, hidden_dim)
        self.init_cell_state = nn.Linear(embed_dim, hidden_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.attention = AttentionModule(encoder_dim=embed_dim)
        self.linear_reshape = nn.Linear(embed_dim + hidden_dim, embed_dim)

    def forward(self, features, captions):
        """
        Forward on
        """
        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_dim).float().to(captions.device))
        embeddings = self.dropout(embeddings)
        seq_len = captions.size(1)
        # get initial hidden_state from image features
        output = torch.empty((captions.size(0), seq_len, self.vocab_size))
        # pass the caption word by word
        hidden_state = self.init_hidden(features.mean(dim=1)).unsqueeze(0)
        cell_state = self.init_cell_state(features.mean(dim=1)).unsqueeze(0)
        for t in range(seq_len):
            # for the first time step the input is the feature vector
            context = self.attention(features, hidden_state)
            concat = torch.concat([context, embeddings[:, t, :]], dim=-1)
            # return to embeddings size after combining attention and context
            lstm_input = self.linear_reshape(concat)
            # use teacher forcing and pass correct answers back
            out, (hidden_state, cell_state) = self.lstm_decoder(lstm_input.unsqueeze(1), (hidden_state, cell_state))

            # output of the rnn, convert to embed dim
            t_out = self.fc_out(out)

            # build the output tensor
            output[:, t, :] = t_out.squeeze()

        return output


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderLSTM(vocab_size, embed_dim, hidden_dim, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
