import torch
import torch.nn as nn
from einops import reduce

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

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

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
                    output_attention=configs.output_attention,
                ) for _ in range(configs.e_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.d_layers)
            ],
        )

        self.transform = Transform(sigma=self.configs.std)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                decomposed=False, attention=False):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc)
        level, growths, seasons, season_attns, growth_attns = self.encoder(res, x_enc, attn_mask=enc_self_mask)

        growth, season, growth_dampings = self.decoder(growths, seasons)

        if decomposed:
            return level[:, -1:], growth, season

        preds = level[:, -1:] + growth + season

        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum('bth,oh->bhot', [growth_attn.squeeze(-1), growth_damping]))

            season_attns = torch.stack(season_attns, dim=0)[:, :, -self.pred_len:]
            season_attns = reduce(season_attns, 'l b d o t -> b o t', reduction='mean')
            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len:]
            decoder_growth_attns = reduce(decoder_growth_attns, 'l b d o t -> b o t', reduction='mean')
            return preds, season_attns, decoder_growth_attns
        return preds
