import torch
import torch.nn as nn
from final_project.src.encoders.encoder_cnn import EncoderCNN
from final_project.src.decoders_w_attention.attention import AttentionModule


class DecoderRNN(nn.Module):
    """
    RNN Decoder without attention
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

    def forward(self, features, captions):
        """
        Forward on
        """
        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_dim).float().to(captions.device))
        embeddings = self.dropout(embeddings)
        # get initial hidden_state from image features
        # pass the caption word by word
        hidden_state = self.init_hidden(features.mean(dim=1)).unsqueeze(0)
        out, hidden_state = self.rnn_decoder(embeddings, hidden_state)
        output = self.fc_out(out)
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
