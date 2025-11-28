import torch
import torch.nn as nn
from final_project.src.encoders.encoder_cnn import EncoderCNN


class AttentionModule(nn.Module):
    """
    Simple Attention implementation from https://github.com/AaronCCWong/Show-Attend-and-Tell/blob/master/attention.py
    """

    def __init__(self, encoder_dim):
        super(AttentionModule, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context


class DecoderRNN(nn.Module):
    """
    RNN Decoder
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn_decoder = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.init_hidden = nn.Linear(embed_dim, hidden_dim)
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
        for t in range(seq_len):
            # for the first time step the input is the feature vector
            context = self.attention(features, hidden_state.squeeze())
            concat = torch.concat([context, embeddings[:, t, :]], dim=-1)
            # return to embeddings size after combining attention and context
            rnn_input = self.linear_reshape(concat)
            # use teacher forcing and pass correct answers back
            out, hidden_state = self.rnn_decoder(rnn_input.unsqueeze(1), hidden_state)

            # output of the rnn, convert to embed dim
            t_out = self.fc_out(out)

            # build the output tensor
            output[:, t, :] = t_out.squeeze()

        return output


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderRNN(vocab_size, embed_dim, hidden_dim, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
