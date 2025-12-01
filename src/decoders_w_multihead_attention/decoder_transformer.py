import torch
import torch.nn as nn
from final_project.src.encoders.encoder_cnn import EncoderCNN


class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        super(DecoderTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._generate_positional_encoding(embed_dim, max_len=5000)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def _generate_positional_encoding(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, features, captions):
        embeddings = self.embedding(captions) * torch.sqrt(torch.tensor(self.embed_dim).float().to(captions.device))
        seq_len = captions.size(1)
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :].to(captions.device)
        embeddings = self.dropout(embeddings)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        output = self.transformer_decoder(tgt=embeddings, memory=features, tgt_mask=tgt_mask)
        prediction = self.fc_out(output)
        return prediction

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderTransformer(vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
