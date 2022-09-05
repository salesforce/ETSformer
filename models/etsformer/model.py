import torch
import torch.nn as nn

from .modules import ETSEmbedding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


class ETSformer(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.configs = configs

        assert configs.d_layers == configs.e_layers

        # Embedding
        self.enc_embedding = ETSEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.seq_len, configs.pred_len, configs.K,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )

        self.transform = Transform(sigma=self.configs.std)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=enc_self_mask)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds
