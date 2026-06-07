import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import time
import random
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================================================
# 1. 基础工具
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

def metric(pred, true, eps=1e-6):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / (np.abs(true) + eps))) * 100
    return mae, rmse, mape

# =========================================================
# 2. 数据集
# =========================================================

class PowerDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, target_idx, c_out):
        super().__init__()

        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.c_out = c_out

        self.length = len(self.data) - self.seq_len - self.pred_len + 1

        if self.length <= 0:
            raise ValueError(
                f"数据长度不足：len(data)={len(self.data)}, "
                f"seq_len={seq_len}, pred_len={pred_len}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        x = self.data[s_begin:s_end]
        y_all = self.data[s_end:s_end + self.pred_len]

        dec_hist = self.data[r_begin:s_end]
        dec_zeros = np.zeros((self.pred_len, self.data.shape[1]), dtype=np.float32)
        dec_inp = np.concatenate([dec_hist, dec_zeros], axis=0)

        if self.c_out == 1:
            y = y_all[:, self.target_idx:self.target_idx + 1]
        else:
            y = y_all[:, :self.c_out]

        return (
            torch.from_numpy(x),
            torch.from_numpy(dec_inp),
            torch.from_numpy(y)
        )

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="gbk")
        except Exception:
            return pd.read_csv(path, encoding="latin1")

def load_power_data(args):
    df = read_csv_safely(args.data_path)
    df.columns = [c.lower().strip() for c in df.columns]

    date_col = args.date_col.lower()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)

    cols = [c.lower().strip() for c in args.cols.split(",")]

    for c in cols:
        if c not in df.columns:
            raise ValueError(f"列 {c} 不存在。当前 CSV 列为：{list(df.columns)}")

    data = df[cols].values.astype(np.float32)
    data = pd.DataFrame(data).interpolate().bfill().ffill().values.astype(np.float32)

    if args.target.lower() == "all":
        args.target_idx = 0
        args.c_out = len(cols)
    else:
        target = args.target.lower()
        if target not in cols:
            raise ValueError(f"target={target} 不在 cols={cols} 中")
        args.target_idx = cols.index(target)
        args.c_out = 1

    args.enc_in = len(cols)

    return data, args

def split_data(data, train_ratio, val_ratio):
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test

def build_dataloaders(args):
    raw_data, args = load_power_data(args)

    train_raw, val_raw, test_raw = split_data(
        raw_data,
        args.train_ratio,
        args.val_ratio
    )

    min_need = args.seq_len + args.pred_len
    if len(train_raw) < min_need or len(val_raw) < min_need or len(test_raw) < min_need:
        raise ValueError(
            f"划分后数据过短，至少需要 {min_need} 条。"
            f"当前 train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}。"
            f"请减小 seq_len/pred_len 或调整 train_ratio/val_ratio。"
        )

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    train_set = PowerDataset(
        train_data, args.seq_len, args.label_len, args.pred_len, args.target_idx, args.c_out
    )
    val_set = PowerDataset(
        val_data, args.seq_len, args.label_len, args.pred_len, args.target_idx, args.c_out
    )
    test_set = PowerDataset(
        test_data, args.seq_len, args.label_len, args.pred_len, args.target_idx, args.c_out
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, scaler, args

def inverse_target(preds, trues, scaler, args):
    if args.c_out == args.enc_in:
        mean = scaler.mean.reshape(1, 1, -1)
        std = scaler.std.reshape(1, 1, -1)
    elif args.c_out == 1:
        mean = scaler.mean[:, args.target_idx].reshape(1, 1, 1)
        std = scaler.std[:, args.target_idx].reshape(1, 1, 1)
    else:
        mean = scaler.mean[:, :args.c_out].reshape(1, 1, -1)
        std = scaler.std[:, :args.c_out].reshape(1, 1, -1)

    return preds * std + mean, trues * std + mean

# =========================================================
# 3. 模型定义
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = self.avg(x_t)
        if out.size(-1) > x_t.size(-1):
            out = out[:, :, :x_t.size(-1)]
        elif out.size(-1) < x_t.size(-1):
            pad_len = x_t.size(-1) - out.size(-1)
            out = F.pad(out, (0, pad_len), mode="replicate")
        return out.transpose(1, 2)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask

# -------------------------
# 3.1 LSTM
# -------------------------

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.c_out = args.c_out

        self.lstm = nn.LSTM(
            input_size=args.enc_in,
            hidden_size=args.d_model,
            num_layers=args.e_layers,
            batch_first=True,
            dropout=args.dropout if args.e_layers > 1 else 0
        )
        self.fc = nn.Linear(args.d_model, args.pred_len * args.c_out)

    def forward(self, x_enc, x_dec=None):
        B = x_enc.size(0)
        out, _ = self.lstm(x_enc)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(B, self.pred_len, self.c_out)
        return out

# -------------------------
# 3.2 TCN
# -------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.c_out = args.c_out

        num_channels = [args.d_model]
        kernel_size = 3

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = args.enc_in if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation

            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    dilation=dilation, padding=padding,
            dropout = args.dropout
            )
            )

            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], args.pred_len * args.c_out)

    def forward(self, x_enc, x_dec=None):
        B = x_enc.size(0)
        x = x_enc.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        out = out.view(B, self.pred_len, self.c_out)
        return out
# -------------------------
# 3.3 Transformer encoder-decoder
# -------------------------

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.c_out = args.c_out

        self.enc_embedding = nn.Linear(args.enc_in, args.d_model)
        self.dec_embedding = nn.Linear(args.enc_in, args.d_model)
        self.pos_encoding = PositionalEncoding(args.d_model)

        self.transformer = nn.Transformer(
            d_model=args.d_model,
            nhead=args.n_heads,
            num_encoder_layers=args.e_layers,
            num_decoder_layers=args.d_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            batch_first=True
        )

        self.projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, x_enc, x_dec=None):
        if x_dec is None:
            raise ValueError("Transformer encoder-decoder 需要 x_dec")

        enc = self.enc_embedding(x_enc)
        enc = self.pos_encoding(enc)

        dec = self.dec_embedding(x_dec)
        dec = self.pos_encoding(dec)

        tgt_mask = generate_square_subsequent_mask(dec.size(1), dec.device)

        out = self.transformer(
            src=enc,
            tgt=dec,
            tgt_mask=tgt_mask
        )

        out = self.projection(out)
        out = out[:, -self.pred_len:, :]
        return out

# -------------------------
# 3.4 Informer encoder-decoder
# ProbSparse Attention
# -------------------------

class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, causal=False):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        index_sample = torch.randint(
            L_K,
            (L_Q, sample_k),
            device=Q.device
        )

        K_sample = K_expand[
            :,
            :,
            torch.arange(L_Q, device=Q.device).unsqueeze(1),
            index_sample,
            :
        ]

        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2),
            K_sample.transpose(-2, -1)
        ).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B, device=Q.device)[:, None, None],
            torch.arange(H, device=Q.device)[None, :, None],
            M_top,
            :
        ]

        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        if not self.causal:
            V_mean = V.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(-1, -1, L_Q, -1).contiguous()
        else:
            context = V.cumsum(dim=-2)
            if L_Q != V.size(-2):
                context = context[:, :, -L_Q:, :]
        return context

    def _update_context(self, context, V, scores_top, index):
        B, H, L_V, D = V.shape

        if self.causal:
            L_Q = context.size(-2)
            mask = torch.ones(
                scores_top.shape,
                dtype=torch.bool,
                device=scores_top.device
            )

            for b in range(B):
                for h in range(H):
                    q_index = index[b, h]
                    key_index = torch.arange(L_V, device=scores_top.device).view(1, -1)
                    mask[b, h] = key_index > q_index.view(-1, 1)

            scores_top = scores_top.masked_fill(mask, -np.inf)

        attn = torch.softmax(scores_top * self.scale, dim=-1)
        attn = self.dropout(attn)

        context[
            torch.arange(context.shape[0], device=context.device)[:, None, None],
            torch.arange(context.shape[1], device=context.device)[None, :, None],
            index,
            :
        ] = torch.matmul(attn, V)

        return context

    def forward(self, Q, K, V):
        B, L_Q, D = Q.shape
        _, L_K, _ = K.shape

        Q = Q.view(B, L_Q, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L_K, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L_K, self.n_heads, self.d_head).transpose(1, 2)

        U_part = max(1, min(L_K, int(self.factor * np.ceil(np.log(L_K + 1)))))
        u = max(1, min(L_Q, int(self.factor * np.ceil(np.log(L_Q + 1)))))

        scores_top, index = self._prob_QK(Q, K, sample_k=U_part, n_top=u)

        context = self._get_initial_context(V, L_Q)
        context = self._update_context(context, V, scores_top, index)

        out = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        Q = self.query_projection(queries)
        K = self.key_projection(keys)
        V = self.value_projection(values)

        out = self.inner_attention(Q, K, V)
        out = self.out_projection(out)
        return self.dropout(out)

class InformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attn = AttentionLayer(
            ProbAttention(
                args.d_model,
                args.n_heads,
                args.dropout,
                factor=args.informer_factor,
                causal=False
            ),
            args.d_model,
            args.n_heads,
            args.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class InformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.self_attn = AttentionLayer(
            ProbAttention(
                args.d_model,
                args.n_heads,
                args.dropout,
                factor=args.informer_factor,
                causal=True
            ),
            args.d_model,
            args.n_heads,
            args.dropout
        )

        self.cross_attn = AttentionLayer(
            ProbAttention(
                args.d_model,
                args.n_heads,
                args.dropout,
                factor=args.informer_factor,
                causal=False
            ),
            args.d_model,
            args.n_heads,
            args.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.norm3 = nn.LayerNorm(args.d_model)

    def forward(self, x, memory):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.cross_attn(x, memory, memory))
        x = self.norm3(x + self.ffn(x))
        return x

class ConvDistillLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.down_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x

class Informer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.c_out = args.c_out
        self.distil = args.informer_distil

        self.enc_embedding = nn.Linear(args.enc_in, args.d_model)
        self.dec_embedding = nn.Linear(args.enc_in, args.d_model)

        self.enc_pos = PositionalEncoding(args.d_model)
        self.dec_pos = PositionalEncoding(args.d_model)

        self.encoder = nn.ModuleList([
            InformerEncoderLayer(args) for _ in range(args.e_layers)
        ])

        self.decoder = nn.ModuleList([
            InformerDecoderLayer(args) for _ in range(args.d_layers)
        ])

        if self.distil and args.e_layers > 1:
            self.distil_layers = nn.ModuleList([
                ConvDistillLayer(args.d_model) for _ in range(args.e_layers - 1)
            ])
        else:
            self.distil_layers = None

        self.projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, x_enc, x_dec=None):
        if x_dec is None:
            raise ValueError("Informer encoder-decoder 需要 x_dec")

        enc = self.enc_embedding(x_enc)
        enc = self.enc_pos(enc)

        if self.distil and self.distil_layers is not None:
            for i, layer in enumerate(self.encoder):
                enc = layer(enc)
                if i < len(self.distil_layers):
                    enc = self.distil_layers[i](enc)
        else:
            for layer in self.encoder:
                enc = layer(enc)

        dec = self.dec_embedding(x_dec)
        dec = self.dec_pos(dec)

        for layer in self.decoder:
            dec = layer(dec, enc)

        out = self.projection(dec)
        out = out[:, -self.pred_len:, :]
        return out

# -------------------------
# 3.5 Autoformer encoder-decoder
# SeriesDecomp + AutoCorrelation
# -------------------------

class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=1.0):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _batch_roll(self, x, shifts):
        # x: [B,H,L,D]
        B, H, L, D = x.shape
        out = []
        for b in range(B):
            out.append(torch.roll(x[b], shifts=-int(shifts[b].item()), dims=-2))
        return torch.stack(out, dim=0)

    def time_delay_agg(self, values, corr):
        # values: [B,H,L,D]
        # corr:   [B,H,L]
        B, H, L, D = values.shape
        top_k = max(1, int(self.factor * math.log(L + 1)))

        mean_corr = corr.mean(dim=1)  # [B,L]
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        out = torch.zeros_like(values)
        for i in range(top_k):
            pattern = self._batch_roll(values, delays[:, i])
            out = out + weights[:, i].view(B, 1, 1, 1) * pattern

        return out

    def forward(self, queries, keys, values):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        q = self.q_proj(queries).view(B, L_Q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(keys).view(B, L_K, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(values).view(B, L_K, self.n_heads, self.d_head).transpose(1, 2)

        # 为了做频域相关，长度统一到 L_Q
        if L_K != L_Q:
            if L_K > L_Q:
                k = k[:, :, -L_Q:, :]
                v = v[:, :, -L_Q:, :]
            else:
                pad_len = L_Q - L_K
                k_pad = k[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                v_pad = v[:, :, -1:, :].repeat(1, 1, pad_len, 1)
                k = torch.cat([k, k_pad], dim=2)
                v = torch.cat([v, v_pad], dim=2)

        q_fft = torch.fft.rfft(q.permute(0, 1, 3, 2), dim=-1)
        k_fft = torch.fft.rfft(k.permute(0, 1, 3, 2), dim=-1)

        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L_Q, dim=-1)  # [B,H,Dh,L]
        corr = corr.mean(dim=2)  # [B,H,L]

        out = self.time_delay_agg(v, corr)  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L_Q, D)
        out = self.o_proj(out)
        out = self.dropout(out)

        return out

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attn = AutoCorrelation(
            args.d_model,
            args.n_heads,
            args.dropout,
            factor=args.autoformer_factor
        )

        self.decomp1 = SeriesDecomp(args.moving_avg)
        self.decomp2 = SeriesDecomp(args.moving_avg)

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        new_x = self.attn(x, x, x)
        x = x + new_x
        x, _ = self.decomp1(x)

        y = self.ffn(x)
        x = x + y
        x, _ = self.decomp2(x)

        x = self.norm(x)
        return x

class AutoformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.self_attn = AutoCorrelation(
            args.d_model,
            args.n_heads,
            args.dropout,
            factor=args.autoformer_factor
        )

        self.cross_attn = AutoCorrelation(
            args.d_model,
            args.n_heads,
            args.dropout,
            factor=args.autoformer_factor
        )

        self.decomp1 = SeriesDecomp(args.moving_avg)
        self.decomp2 = SeriesDecomp(args.moving_avg)
        self.decomp3 = SeriesDecomp(args.moving_avg)

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x, memory):
        x = x + self.self_attn(x, x, x)
        x, _ = self.decomp1(x)

        x = x + self.cross_attn(x, memory, memory)
        x, _ = self.decomp2(x)

        x = x + self.ffn(x)
        x, _ = self.decomp3(x)

        x = self.norm(x)
        return x

class Autoformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.c_out = args.c_out

        self.enc_embedding = nn.Linear(args.enc_in, args.d_model)
        self.dec_embedding = nn.Linear(args.enc_in, args.d_model)

        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(args) for _ in range(args.e_layers)
        ])

        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(args) for _ in range(args.d_layers)
        ])

        self.projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, x_enc, x_dec=None):
        if x_dec is None:
            raise ValueError("Autoformer encoder-decoder 需要 x_dec")

        enc = self.enc_embedding(x_enc)

        for layer in self.encoder:
            enc = layer(enc)

        dec = self.dec_embedding(x_dec)

        for layer in self.decoder:
            dec = layer(dec, enc)

        out = self.projection(dec)
        out = out[:, -self.pred_len:, :]
        return out

# -------------------------
# 3.6 FEDformer encoder-decoder
# Fourier Enhanced Block
# -------------------------

class FourierBlock(nn.Module):
    def __init__(self, d_model, n_heads, modes=32, mode_select_method="low"):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = modes
        self.mode_select_method = mode_select_method

        scale = 1.0 / (d_model * d_model)

        self.weights_real = nn.Parameter(
            scale * torch.randn(n_heads, self.d_head, self.d_head, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(n_heads, self.d_head, self.d_head, modes)
        )

    def _get_frequency_modes(self, seq_len, modes):
        max_modes = seq_len // 2 + 1
        modes = min(modes, max_modes)

        if self.mode_select_method == "random":
            index = list(np.random.choice(max_modes, modes, replace=False))
            index.sort()
        else:
            index = list(range(modes))

        return index

    def compl_mul1d(self, input_ft, weights_real, weights_imag):
        weights = torch.complex(weights_real, weights_imag)
        return torch.einsum("bhdm,hedm->bhem", input_ft, weights)

    def forward(self, x):
        # x: [B,L,D]
        B, L, D = x.shape

        x = x.view(B, L, self.n_heads, self.d_head)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,Dh,L]

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            B,
            self.n_heads,
            self.d_head,
            L // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )

        index = self._get_frequency_modes(L, self.modes)

        for wi, i in enumerate(index):
            out_ft[:, :, :, i] = self.compl_mul1d(
                x_ft[:, :, :, i:i + 1],
                self.weights_real[:, :, :, wi:wi + 1],
                self.weights_imag[:, :, :, wi:wi + 1]
            ).squeeze(-1)

        x = torch.fft.irfft(out_ft, n=L, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, L, D)

        return x

class FourierCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes=32, mode_select_method="low"):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = modes
        self.mode_select_method = mode_select_method

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _get_frequency_modes(self, seq_len, modes):
        max_modes = seq_len // 2 + 1
        modes = min(modes, max_modes)

        if self.mode_select_method == "random":
            index = list(np.random.choice(max_modes, modes, replace=False))
            index.sort()
        else:
            index = list(range(modes))

        return index

    def forward(self, queries, keys, values):
        # 这里保留 FEDformer 的 Fourier cross 思路：
        # q/k/v 投影后在频域低频 mode 上做增强，再回到时域。
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        if L_K != L_Q:
            if L_K > L_Q:
                k = k[:, -L_Q:, :]
                v = v[:, -L_Q:, :]
            else:
                pad_len = L_Q - L_K
                k = torch.cat([k, k[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
                v = torch.cat([v, v[:, -1:, :].repeat(1, pad_len, 1)], dim=1)

        q = q.view(B, L_Q, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        k = k.view(B, L_Q, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, L_Q, self.n_heads, self.d_head).permute(0, 2, 3, 1)

        q_ft = torch.fft.rfft(q, dim=-1)
        k_ft = torch.fft.rfft(k, dim=-1)
        v_ft = torch.fft.rfft(v, dim=-1)

        out_ft = torch.zeros_like(v_ft)

        index = self._get_frequency_modes(L_Q, self.modes)

        for i in index:
            score = q_ft[:, :, :, i] * torch.conj(k_ft[:, :, :, i])
            score = torch.softmax(score.abs(), dim=-1).type_as(v_ft)
            out_ft[:, :, :, i] = score * v_ft[:, :, :, i]

        out = torch.fft.irfft(out_ft, n=L_Q, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L_Q, D)
        out = self.o_proj(out)

        return out

class FEDformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fourier = FourierBlock(
            d_model=args.d_model,
            n_heads=args.n_heads,
            modes=args.fedformer_modes,
            mode_select_method=args.mode_select
        )

        self.decomp1 = SeriesDecomp(args.moving_avg)
        self.decomp2 = SeriesDecomp(args.moving_avg)

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        x = x + self.fourier(x)
        x, _ = self.decomp1(x)

        x = x + self.ffn(x)
        x, _ = self.decomp2(x)

        x = self.norm(x)
        return x

class FEDformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.self_fourier = FourierBlock(
            d_model=args.d_model,
            n_heads=args.n_heads,
            modes=args.fedformer_modes,
            mode_select_method=args.mode_select
        )

        self.cross_fourier = FourierCrossAttention(
            d_model=args.d_model,
            n_heads=args.n_heads,
            modes=args.fedformer_modes,
            mode_select_method=args.mode_select
        )

        self.decomp1 = SeriesDecomp(args.moving_avg)
        self.decomp2 = SeriesDecomp(args.moving_avg)
        self.decomp3 = SeriesDecomp(args.moving_avg)

        self.ffn = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_ff, args.d_model),
            nn.Dropout(args.dropout)
        )

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x, memory):
        x = x + self.self_fourier(x)
        x, _ = self.decomp1(x)

        x = x + self.cross_fourier(x, memory, memory)
        x, _ = self.decomp2(x)

        x = x + self.ffn(x)
        x, _ = self.decomp3(x)

        x = self.norm(x)
        return x

class FEDformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.c_out = args.c_out

        self.enc_embedding = nn.Linear(args.enc_in, args.d_model)
        self.dec_embedding = nn.Linear(args.enc_in, args.d_model)

        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(args) for _ in range(args.e_layers)
        ])

        self.decoder = nn.ModuleList([
            FEDformerDecoderLayer(args) for _ in range(args.d_layers)
        ])

        self.projection = nn.Linear(args.d_model, args.c_out)

    def forward(self, x_enc, x_dec=None):
        if x_dec is None:
            raise ValueError("FEDformer encoder-decoder 需要 x_dec")

        enc = self.enc_embedding(x_enc)

        for layer in self.encoder:
            enc = layer(enc)

        dec = self.dec_embedding(x_dec)

        for layer in self.decoder:
            dec = layer(dec, enc)

        out = self.projection(dec)
        out = out[:, -self.pred_len:, :]
        return out

# -------------------------
# 3.7 PatchTST
# -------------------------

class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.c_out = args.c_out

        self.patch_len = args.patch_len
        self.stride = args.patch_stride

        if args.seq_len < self.patch_len:
            raise ValueError("seq_len 必须大于等于 patch_len")

        self.num_patches = (args.seq_len - self.patch_len) // self.stride + 1

        self.patch_embedding = nn.Linear(self.patch_len * args.enc_in, args.d_model)
        self.pos_encoding = PositionalEncoding(args.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.e_layers
        )

        self.fc = nn.Linear(args.d_model, args.pred_len * args.c_out)

    def forward(self, x_enc, x_dec=None):
        B, L, C = x_enc.shape

        patches = []

        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end > L:
                break

            patch = x_enc[:, start:end, :].reshape(B, -1)
            patches.append(patch)

        patches = torch.stack(patches, dim=1)

        x = self.patch_embedding(patches)
        x = self.pos_encoding(x)

        out = self.transformer_encoder(x)
        out = out[:, -1, :]

        out = self.fc(out)
        out = out.view(B, self.pred_len, self.c_out)

        return out

# -------------------------
# 3.8 DLinear 弱化版
# -------------------------

class DLinear(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.c_out = args.c_out
        self.target_idx = args.target_idx

        # 弱化：小分解核 + 通道共享，不做变量间联合建模
        self.decomp = SeriesDecomp(kernel_size=3)

        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len, bias=True)

        nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
        nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)

        nn.init.zeros_(self.linear_seasonal.bias)
        nn.init.zeros_(self.linear_trend.bias)

    def _select_output(self, y_all):
        if self.c_out == self.enc_in:
            return y_all
        elif self.c_out == 1:
            return y_all[:, :, self.target_idx:self.target_idx + 1]
        else:
            return y_all[:, :, :self.c_out]

    def forward(self, x_enc, x_dec=None):
        seasonal, trend = self.decomp(x_enc)

        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)

        out = seasonal_out + trend_out
        out = out.transpose(1, 2)

        out = self._select_output(out)
        return out

# -------------------------
# 3.9 iTransformer
# -------------------------

class iTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.c_out = args.c_out
        self.target_idx = args.target_idx

        self.var_embedding = nn.Linear(args.seq_len, args.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.e_layers
        )

        self.fc = nn.Linear(args.d_model, args.pred_len)

    def forward(self, x_enc, x_dec=None):
        x = x_enc.transpose(1, 2)
        x = self.var_embedding(x)

        out = self.transformer_encoder(x)
        out = self.fc(out)
        out = out.transpose(1, 2)

        if self.c_out == 1:
            out = out[:, :, self.target_idx:self.target_idx + 1]
        elif self.c_out != self.enc_in:
            out = out[:, :, :self.c_out]

        return out

# -------------------------
# 3.10 TimesNet
# -------------------------

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()

        kernels = []
        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i
                )
            )

        self.kernels = nn.ModuleList(kernels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        res_list = []

        for kernel in self.kernels:
            res_list.append(kernel(x))

        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.k = args.top_k_period

        self.conv = nn.Sequential(
            Inception_Block_V1(args.enc_in, args.d_model, num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(args.d_model, args.enc_in, num_kernels=6)
        )

    def FFT_for_Period(self, x, k=5):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = torch.abs(xf).mean(0).mean(-1)

        if frequency_list.numel() > 0:
            frequency_list[0] = 0

        if frequency_list.shape[0] <= 1:
            top_list = torch.tensor([1], device=x.device, dtype=torch.long)
        else:
            real_k = min(k, frequency_list.shape[0] - 1)
            _, top_list = torch.topk(frequency_list, real_k)
            top_list = torch.clamp(top_list, min=1)

        period = x.shape[1] // top_list
        period = torch.clamp(period, min=1)

        period_weight = torch.abs(xf).mean(-1)[:, top_list]

        return period.detach().cpu().numpy(), period_weight

    def forward(self, x):
        B, L, C = x.shape

        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        real_k = len(period_list)

        for i in range(real_k):
            period = int(period_list[i])
            period = max(period, 1)

            if (L % period) != 0:
                length = ((L // period) + 1) * period
                padding = torch.zeros(
                    [B, length - L, C],
                    device=x.device,
                    dtype=x.dtype
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = L
                out = x

            out = out.reshape(B, length // period, period, C)
            out = out.permute(0, 3, 1, 2).contiguous()

            out = self.conv(out)

            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :L, :])

        res = torch.stack(res, dim=-1)

        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        period_weight = period_weight.repeat(1, L, C, 1)

        res = torch.sum(res * period_weight, dim=-1)
        res = res + x

        return res

class TimesNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.c_out = args.c_out
        self.target_idx = args.target_idx

        self.model = nn.ModuleList([
            TimesBlock(args) for _ in range(args.e_layers)
        ])

        self.layer_norm = nn.LayerNorm(args.enc_in)
        self.fc = nn.Linear(args.seq_len, args.pred_len)

    def forward(self, x_enc, x_dec=None):
        x = x_enc

        for layer in self.model:
            x = layer(x)

        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        out = self.fc(x)
        out = out.transpose(1, 2)

        if self.c_out == 1:
            out = out[:, :, self.target_idx:self.target_idx + 1]
        elif self.c_out != self.enc_in:
            out = out[:, :, :self.c_out]

        return out

# -------------------------
# 3.11 TimeMixer
# -------------------------

class TimeMixerMovingAvg(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t)

        if trend.size(-1) > x_t.size(-1):
            trend = trend[:, :, :x_t.size(-1)]
        elif trend.size(-1) < x_t.size(-1):
            pad_len = x_t.size(-1) - trend.size(-1)
            trend = F.pad(trend, (0, pad_len), mode="replicate")

        trend = trend.transpose(1, 2)
        return trend

class TimeMixerBlock(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout=0.1, kernel_size=25):
        super().__init__()

        self.seq_len = seq_len
        self.enc_in = enc_in

        self.decomp = TimeMixerMovingAvg(kernel_size=kernel_size)

        self.seasonal_temporal_mixer = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )

        self.trend_temporal_mixer = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )

        self.channel_norm = nn.LayerNorm(enc_in)

        self.channel_mixer = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout)
        )

        self.out_norm = nn.LayerNorm(enc_in)

    def forward(self, x):
        residual = x

        trend = self.decomp(x)
        seasonal = x - trend

        seasonal_t = seasonal.transpose(1, 2)
        trend_t = trend.transpose(1, 2)

        seasonal_out = self.seasonal_temporal_mixer(seasonal_t)
        trend_out = self.trend_temporal_mixer(trend_t)

        seasonal_out = seasonal_out.transpose(1, 2)
        trend_out = trend_out.transpose(1, 2)

        x = seasonal_out + trend_out
        x = x + self.channel_mixer(self.channel_norm(x))
        x = self.out_norm(x + residual)

        return x

class TimeMixer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.c_out = args.c_out
        self.target_idx = args.target_idx

        self.d_model = args.d_model
        self.dropout = args.dropout

        self.num_scales = getattr(args, "timemixer_scales", 3)
        self.kernel_size = getattr(args, "timemixer_kernel", 25)

        scale_lens = []
        cur_len = args.seq_len

        for _ in range(self.num_scales):
            if cur_len < 2:
                break
            scale_lens.append(cur_len)
            cur_len = cur_len // 2

        self.scale_lens = scale_lens
        self.real_scales = len(scale_lens)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TimeMixerBlock(
                    seq_len=scale_len,
                    enc_in=args.enc_in,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    kernel_size=self.kernel_size
                )
                for _ in range(args.e_layers)
            ])
            for scale_len in self.scale_lens
        ])

        self.pred_heads = nn.ModuleList([
            nn.Linear(scale_len, args.pred_len)
            for scale_len in self.scale_lens
        ])

        self.scale_weights = nn.Parameter(torch.zeros(self.real_scales))

    def _downsample(self, x):
        x_t = x.transpose(1, 2)
        x_t = F.avg_pool1d(
            x_t,
            kernel_size=2,
            stride=2,
            ceil_mode=False
        )
        return x_t.transpose(1, 2)

    def _select_output(self, y_all):
        if self.c_out == self.enc_in:
            return y_all
        elif self.c_out == 1:
            return y_all[:, :, self.target_idx:self.target_idx + 1]
        else:
            return y_all[:, :, :self.c_out]

    def forward(self, x_enc, x_dec=None):
        x = x_enc
        B, L, C = x.shape

        xs = [x]
        cur = x

        for i in range(1, self.real_scales):
            cur = self._downsample(cur)
            expected_len = self.scale_lens[i]

            if cur.size(1) > expected_len:
                cur = cur[:, :expected_len, :]
            elif cur.size(1) < expected_len:
                pad_len = expected_len - cur.size(1)
                pad = cur[:, -1:, :].repeat(1, pad_len, 1)
                cur = torch.cat([cur, pad], dim=1)

            xs.append(cur)

        preds = []

        for i in range(self.real_scales):
            h = xs[i]

            for block in self.blocks[i]:
                h = block(h)

            h_t = h.transpose(1, 2)
            y = self.pred_heads[i](h_t)
            y = y.transpose(1, 2)

            preds.append(y)

        weights = F.softmax(self.scale_weights, dim=0)

        y_all = 0.0

        for i in range(self.real_scales):
            y_all = y_all + weights[i] * preds[i]

        y = self._select_output(y_all)

        return y

# =========================================================
# 4. 模型工厂
# =========================================================

def get_model(args):
    model_name = args.model.lower()

    if model_name == "lstm":
        return LSTM(args)
    elif model_name == "tcn":
        return TCN(args)
    elif model_name == "transformer":
        return Transformer(args)
    elif model_name == "informer":
        return Informer(args)
    elif model_name == "autoformer":
        return Autoformer(args)
    elif model_name == "fedformer":
        return FEDformer(args)
    elif model_name == "patchtst":
        return PatchTST(args)
    elif model_name == "dlinear":
        return DLinear(args)
    elif model_name == "itransformer":
        return iTransformer(args)
    elif model_name == "timesnet":
        return TimesNet(args)
    elif model_name == "timemixer":
        return TimeMixer(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

# =========================================================
# 5. 训练 / 验证 / 测试
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()

    total = 0.0
    count = 0

    for batch in loader:
        if len(batch) == 3:
            x_enc, x_dec, y = batch
        else:
            raise ValueError("当前 DataLoader 应返回 x_enc, x_dec, y 三项。")

        x_enc = x_enc.float().to(device)
        x_dec = x_dec.float().to(device)
        y = y.float().to(device)

        optimizer.zero_grad()

        pred = model(x_enc, x_dec)
        loss = criterion(pred, y)

        loss.backward()

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        total += loss.item()
        count += 1

    return total / max(count, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total = 0.0
    count = 0

    preds = []
    trues = []

    for batch in loader:
        if len(batch) == 3:
            x_enc, x_dec, y = batch
        else:
            raise ValueError("当前 DataLoader 应返回 x_enc, x_dec, y 三项。")

        x_enc = x_enc.float().to(device)
        x_dec = x_dec.float().to(device)
        y = y.float().to(device)

        pred = model(x_enc, x_dec)
        loss = criterion(pred, y)

        total += loss.item()
        count += 1

        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    if len(preds) == 0:
        raise ValueError("evaluate 阶段没有有效 batch，请检查数据长度或 batch_size。")

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae, rmse, mape = metric(preds, trues)

    return total / max(count, 1), mae, rmse, mape, preds, trues

def run_single_experiment(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )

    train_loader, val_loader, test_loader, scaler, args = build_dataloaders(args)

    model = get_model(args).to(device)

    param_count = count_parameters(model)
    params_M = param_count / 1e6
    model_size_MB = param_count * 4 / (1024 ** 2)

    if args.loss == "mae":
        criterion = nn.L1Loss().to(device)
    elif args.loss == "mse":
        criterion = nn.MSELoss().to(device)
    elif args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta).to(device)
    else:
        raise ValueError(f"未知 loss: {args.loss}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience
    )

    save_dir = os.path.join(args.checkpoints, args.model_id)
    result_dir = os.path.join(args.results, args.model_id)

    ensure_dir(save_dir)
    ensure_dir(result_dir)

    best_path = os.path.join(save_dir, "best_model.pth")

    best_val = float("inf")
    patience_count = 0

    train_losses = []
    val_losses = []

    start_time = time.time()

    print("=" * 100)
    print(f"Experiment: {args.model_id}")
    print(f"Model     : {args.model}")
    print(f"Data      : {args.data_path}")
    print(f"Target    : {args.target}")
    print(f"seq_len   : {args.seq_len}")
    print(f"label_len : {args.label_len}")
    print(f"pred_len  : {args.pred_len}")
    print(f"d_model   : {args.d_model}")
    print(f"Device    : {device}")
    print(f"Loss      : {args.loss}")
    print(f"Params    : {params_M:.4f} M")
    print(f"ModelSize : {model_size_MB:.4f} MB")
    print("=" * 100)

    for epoch in range(1, args.train_epochs + 1):
        ep_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args
        )

        val_loss, val_mae, val_rmse, val_mape, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:03d}/{args.train_epochs}] | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"Val MAE {val_mae:.6f} | "
            f"Val RMSE {val_rmse:.6f} | "
            f"Val MAPE {val_mape:.3f}% | "
            f"Time {time.time() - ep_start:.2f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "train_losses": train_losses,
                    "val_losses": val_losses
                },
                best_path
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if device.type == "cuda":
        torch.cuda.synchronize()

    test_start_time = time.time()

    test_loss, test_mae, test_rmse, test_mape, preds, trues = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    test_time_sec = time.time() - test_start_time
    num_test_samples = preds.shape[0]
    inference_time_per_sample_ms = test_time_sec / max(num_test_samples, 1) * 1000.0

    preds_inv, trues_inv = inverse_target(preds, trues, scaler, args)
    mae_inv, rmse_inv, mape_inv = metric(preds_inv, trues_inv)

    np.save(os.path.join(result_dir, "pred.npy"), preds_inv)
    np.save(os.path.join(result_dir, "true.npy"), trues_inv)

    metrics_path = os.path.join(result_dir, "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"model_id: {args.model_id}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"target: {args.target}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"label_len: {args.label_len}\n")
        f.write(f"pred_len: {args.pred_len}\n")
        f.write(f"d_model: {args.d_model}\n")
        f.write(f"n_heads: {args.n_heads}\n")
        f.write(f"e_layers: {args.e_layers}\n")
        f.write(f"d_layers: {args.d_layers}\n")
        f.write(f"d_ff: {args.d_ff}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"params: {param_count}\n")
        f.write(f"params_M: {params_M:.6f}\n")
        f.write(f"model_size_MB: {model_size_MB:.6f}\n")
        f.write(f"train_time_sec: {train_time:.2f}\n")
        f.write(f"test_time_sec: {test_time_sec:.6f}\n")
        f.write(f"inference_time_per_sample_ms: {inference_time_per_sample_ms:.6f}\n")
        f.write("\n")
        f.write("Normalized scale:\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"mae: {test_mae:.6f}\n")
        f.write(f"rmse: {test_rmse:.6f}\n")
        f.write(f"mape: {test_mape:.6f}\n")
        f.write("\n")
        f.write("Original scale:\n")
        f.write(f"mae: {mae_inv:.6f}\n")
        f.write(f"rmse: {rmse_inv:.6f}\n")
        f.write(f"mape: {mape_inv:.6f}\n")

    print("-" * 100)
    print(f"Finished: {args.model_id}")
    print(f"Original scale MAE={mae_inv:.6f}, RMSE={rmse_inv:.6f}, MAPE={mape_inv:.3f}%")
    print(f"Params: {param_count} | Params_M: {params_M:.6f} M | ModelSize: {model_size_MB:.6f} MB")
    print(
        f"Train time: {train_time:.2f}s | "
        f"Test time: {test_time_sec:.6f}s | "
        f"Inference/sample: {inference_time_per_sample_ms:.6f} ms"
    )
    print(f"Model saved to: {best_path}")
    print("-" * 100)

    result = {
        "model_id": args.model_id,
        "model": args.model,
        "data_path": args.data_path,
        "target": args.target,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "loss": args.loss,
        "params": param_count,
        "params_M": params_M,
        "model_size_MB": model_size_MB,
        "test_loss_norm": test_loss,
        "mae_norm": test_mae,
        "rmse_norm": test_rmse,
        "mape_norm": test_mape,
        "mae": mae_inv,
        "rmse": rmse_inv,
        "mape": mape_inv,
        "train_time_sec": train_time,
        "test_time_sec": test_time_sec,
        "inference_time_per_sample_ms": inference_time_per_sample_ms,
        "checkpoint": best_path,
    }

    return result

# =========================================================
# 6. 参数
# =========================================================

def get_args():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="GEF")
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--cols", type=str, default="load,solar,wind")
    parser.add_argument("--target", type=str, default="all")

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # 长度
    parser.add_argument("--seq_len", type=int, default=240)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)

    # 模型
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "LSTM",
            "TCN",
            "Transformer",
            "Informer",
            "Autoformer",
            "FEDformer",
            "PatchTST",
            "DLinear",
            "iTransformer",
            "TimesNet",
            "TimeMixer"
        ]
    )

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # PatchTST
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)

    # Informer
    parser.add_argument("--informer_factor", type=int, default=5)
    parser.add_argument("--informer_distil", action="store_true")

    # Autoformer / FEDformer
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--autoformer_factor", type=float, default=1.0)

    # FEDformer
    parser.add_argument("--fedformer_modes", type=int, default=32)
    parser.add_argument("--mode_select", type=str, default="low", choices=["low", "random"])

    # TimesNet
    parser.add_argument("--top_k_period", type=int, default=5)

    # TimeMixer
    parser.add_argument("--timemixer_scales", type=int, default=3)
    parser.add_argument("--timemixer_kernel", type=int, default=25)

    # 自动由数据推断
    parser.add_argument("--enc_in", type=int, default=3)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--target_idx", type=int, default=0)

    # 训练
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # loss
    parser.add_argument(
        "--loss",
        type=str,
        default="mae",
        choices=["mae", "mse", "huber"]
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)

    # 系统
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--results", type=str, default="./results")

    args = parser.parse_args()

    if args.label_len > args.seq_len:
        raise ValueError(
            f"label_len 不能大于 seq_len，当前 label_len={args.label_len}, seq_len={args.seq_len}"
        )

    if args.model_id == "":
        args.model_id = f"{args.data_name}_{args.target}_{args.model}_seed{args.seed}"

    return args

if __name__ == "__main__":
    args = get_args()

    result = run_single_experiment(args)

    ensure_dir(args.results)
    summary_path = os.path.join(args.results, "summary_baselines.csv")

    result_df = pd.DataFrame([result])

    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        new_df = pd.concat([old_df, result_df], ignore_index=True)
    else:
        new_df = result_df

    new_df.to_csv(summary_path, index=False)

    print(f"\nResult appended to {summary_path}")