import torch
import torch.nn as nn
from final_project.src.encoders.encoder_cnn import EncoderCNN


class DecoderRNN(nn.Module):
    """
    RNN Decoder
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_heads):
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
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True,
                                               dropout=dropout)
        self.linear_reshape = nn.Linear(embed_dim + hidden_dim, embed_dim)

    def forward(self, features, captions):
        """
        Forward on
        """
        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_dim).float().to(captions.device))
        embeddings = self.dropout(embeddings)
        # pass the caption word by word
        hidden_state = self.init_hidden(features.mean(dim=1)).unsqueeze(0)
        attention, weights = self.attention(query=embeddings, key=features, value=features)
        concat = torch.concat([embeddings, attention], dim=-1)
        rnn_input = self.linear_reshape(concat)
        out, hidden_state = self.rnn_decoder(rnn_input, hidden_state)
        output = self.fc_out(out)
        return output


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_heads):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderRNN(vocab_size, embed_dim, hidden_dim, dropout, num_heads)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
