import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class MLPModel(nn.Module):
    def __init__(self, input_size, seq_len, hidden_sizes=None, dropout=0.2, pred_len=1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        dims = [input_size * seq_len, *hidden_sizes]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-1], pred_len))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


class VanillaLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, pred_len=1, bidirectional=False):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(enc_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        pred_len=1,
        bidirectional=False,
        attention_pool=False,
        input_proj_size=None,
        ar_window=0,
        context_window=12,
        feature_names=None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.bidirectional = bool(bidirectional)
        self.attention_pool = bool(attention_pool)
        self.ar_window = max(0, int(ar_window))
        self.context_window = max(2, int(context_window))
        self.feature_names = list(feature_names or [])
        self.target_idx = self.feature_names.index("LogCn2") if "LogCn2" in self.feature_names else 0
        proj_size = int(input_proj_size or input_size)

        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, proj_size) if proj_size != input_size else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        enc_size = hidden_size * (2 if self.bidirectional else 1)
        if self.attention_pool:
            self.temporal_attention = nn.Sequential(
                nn.Linear(enc_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.temporal_attention = None

        stat_size = 5
        deep_in = enc_size * 2 + stat_size
        self.deep_head = nn.Sequential(
            nn.Linear(deep_in, enc_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_size, pred_len),
        )
        self.shortcut_head = nn.Linear(max(1, self.ar_window), pred_len)
        self.mix_gate = nn.Sequential(
            nn.Linear(deep_in, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_len),
            nn.Sigmoid(),
        )

    def _target_context(self, x):
        target_hist = x[:, :, self.target_idx]
        ctx_window = min(self.context_window, target_hist.size(1))
        recent = target_hist[:, -ctx_window:]
        last_val = recent[:, -1:]
        recent_mean = recent.mean(dim=1, keepdim=True)
        recent_std = recent.std(dim=1, keepdim=True, unbiased=False)
        recent_slope = recent[:, -1:] - recent[:, :1]
        last_gap = last_val - recent_mean
        stats = torch.cat([last_val, recent_mean, recent_std, recent_slope, last_gap], dim=1)
        if self.ar_window > 0:
            ar_in = target_hist[:, -min(self.ar_window, target_hist.size(1)) :]
            if ar_in.size(1) < self.ar_window:
                pad = ar_in.new_zeros(ar_in.size(0), self.ar_window - ar_in.size(1))
                ar_in = torch.cat([pad, ar_in], dim=1)
        else:
            ar_in = last_val
        return stats, ar_in

    def _forward_internal(self, x):
        z = self.input_norm(x)
        z = self.input_proj(z)
        out, _ = self.lstm(z)
        last_state = out[:, -1, :]
        if self.temporal_attention is not None:
            attn = torch.softmax(self.temporal_attention(out), dim=1)
            pooled = (attn * out).sum(dim=1)
        else:
            pooled = out.mean(dim=1)
        stats, ar_in = self._target_context(x)
        deep_in = torch.cat([last_state, pooled, stats], dim=1)
        deep_pred = self.deep_head(deep_in)
        shortcut_pred = self.shortcut_head(ar_in)
        gate = self.mix_gate(deep_in)
        pred = gate * deep_pred + (1.0 - gate) * shortcut_pred
        return pred

    def forward(self, x):
        return self._forward_internal(x)

    def compute_loss(self, x, y, loss_cfg=None):
        loss_cfg = loss_cfg or {}
        pred = self._forward_internal(x)
        mse = nn.functional.mse_loss(pred, y)
        huber = nn.functional.smooth_l1_loss(pred, y, beta=float(loss_cfg.get("huber_beta", 0.2)))
        huber_weight = float(loss_cfg.get("huber_weight", 0.35))
        diff_weight = float(loss_cfg.get("diff_weight", 0.1))
        total = mse + huber_weight * huber
        if pred.size(1) > 1 and diff_weight > 0.0:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            true_diff = y[:, 1:] - y[:, :-1]
            total = total + diff_weight * nn.functional.mse_loss(pred_diff, true_diff)
        stats = {
            "mse": float(mse.detach().cpu()),
            "huber": float(huber.detach().cpu()),
        }
        return pred, total, stats


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, pred_len=1):
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.1,
        pred_len=1,
        use_positional_encoding=True,
        max_len=512,
        pooling="last",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pooling = pooling

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):
        z = self.input_proj(x)
        if self.use_positional_encoding:
            if z.size(1) > self.positional_encoding.size(1):
                raise ValueError(
                    f"Sequence length {z.size(1)} exceeds max_len {self.positional_encoding.size(1)} for positional encoding"
                )
            z = z + self.positional_encoding[:, : z.size(1), :]
        z = self.encoder(z)
        if self.pooling == "mean":
            pooled = z.mean(dim=1)
        else:
            pooled = z[:, -1, :]
        return self.fc(pooled)


class PatchTSTModel(nn.Module):
    """Lightweight patch-based transformer baseline for long-horizon sequence modeling."""

    def __init__(
        self,
        input_size,
        patch_len=6,
        stride=3,
        d_model=96,
        nhead=4,
        num_layers=3,
        ff_dim=192,
        dropout=0.1,
        pred_len=1,
        pooling="mean",
        max_patches=128,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pooling = pooling

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.channel_embed = nn.Parameter(torch.zeros(1, self.input_size, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]
        if x.size(-1) < self.patch_len:
            x = F.pad(x, (self.patch_len - x.size(-1), 0))
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        bsz, channels, n_patches, _ = patches.shape
        z = self.patch_embed(patches)  # [B, C, N, D]
        pos = self.pos_embed[:, :n_patches, :].unsqueeze(1)
        z = self.dropout(z + self.channel_embed[:, :channels, :, :] + pos)
        z = z.reshape(bsz * channels, n_patches, -1)
        z = self.encoder(z)
        if self.pooling == "last":
            z = z[:, -1, :]
        else:
            z = z.mean(dim=1)
        z = z.reshape(bsz, channels, -1).mean(dim=1)
        z = self.norm(z)
        return self.head(z)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        output_dim = int(output_dim or input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        gated = self.gate(out) * out
        return self.norm(gated + self.skip(x))


class TFTModel(nn.Module):
    """Lightweight Temporal Fusion Transformer-style baseline."""

    def __init__(
        self,
        input_size,
        hidden_size=96,
        lstm_layers=1,
        num_heads=4,
        dropout=0.1,
        pred_len=1,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.value_proj = nn.Linear(1, hidden_size)
        self.var_select = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
        )
        self.local_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, x):
        # x: [B, L, C]
        weights = torch.softmax(self.var_select(x), dim=-1)
        values = self.value_proj(x.unsqueeze(-1))  # [B, L, C, H]
        selected = (weights.unsqueeze(-1) * values).sum(dim=2)  # [B, L, H]
        local = self.local_grn(selected)
        enc, _ = self.encoder(local)
        attn_out, _ = self.self_attn(enc, enc, enc, need_weights=False)
        fused = self.attn_grn(attn_out + enc)
        last_context = fused[:, -1, :]
        global_context = fused.mean(dim=1)
        gate = self.context_gate(torch.cat([last_context, global_context], dim=-1))
        context = gate * last_context + (1.0 - gate) * global_context
        return self.head(context)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation,
            )
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        if out.size(2) > x.size(2):
            out = out[:, :, -x.size(2) :]
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) > out.size(2):
            res = res[:, :, -out.size(2) :]
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size, channels=None, kernel_size=3, dropout=0.2, pred_len=1):
        super().__init__()
        if channels is None:
            channels = [64, 64]
        layers = []
        for i, out_c in enumerate(channels):
            in_c = input_size if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_c, out_c, kernel_size, 1, 2**i, dropout))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], pred_len)

    def forward(self, x):
        z = x.permute(0, 2, 1)
        z = self.net(z)
        return self.fc(z[:, :, -1])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.2, pred_len=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout / 2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.Dropout(dropout / 2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, pred_len),
        )

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.cnn(z).transpose(1, 2)
        z, _ = self.lstm(z)
        return self.fc(z[:, -1, :])


class RTDMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, regime_size: int, physics_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.regime_size = regime_size
        fusion_in = input_size + hidden_size + physics_size

        self.regime_net = nn.Sequential(
            nn.Linear(fusion_in, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, regime_size),
        )
        self.slow_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size + regime_size + physics_size, hidden_size),
            nn.Sigmoid(),
        )
        self.fast_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size + regime_size + physics_size, hidden_size),
            nn.Sigmoid(),
        )
        self.slow_candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size + regime_size + physics_size, hidden_size),
            nn.Tanh(),
        )
        self.fast_candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size + regime_size + physics_size, hidden_size),
            nn.Tanh(),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + regime_size + physics_size, hidden_size),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, slow_prev, fast_prev, mix_prev, phys_t):
        regime_logits = self.regime_net(torch.cat([x_t, mix_prev, phys_t], dim=-1))
        regime = torch.softmax(regime_logits, dim=-1)
        common = torch.cat([x_t, mix_prev, regime, phys_t], dim=-1)

        slow_gate = self.slow_gate(common)
        fast_gate = self.fast_gate(common)
        slow_cand = self.slow_candidate(common)
        fast_cand = self.fast_candidate(common)

        slow = (1.0 - slow_gate) * slow_prev + slow_gate * slow_cand
        fast = (1.0 - fast_gate) * fast_prev + fast_gate * fast_cand

        mix_gate = self.mix_gate(torch.cat([slow, fast, regime, phys_t], dim=-1))
        mixed = mix_gate * slow + (1.0 - mix_gate) * fast
        mixed = self.dropout(mixed)
        return slow, fast, mixed, regime


class RTDMNetModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 96,
        regime_size: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 1,
        feature_names=None,
        stable_branch: bool = False,
        attention_pool: bool = False,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.feature_names = list(feature_names or [])
        self.stable_branch = stable_branch
        self.attention_pool = attention_pool
        preferred = [
            "SolarFlux",
            "SolarFlux_nonneg",
            "Tair_IRtemp_gap",
            "UmSonic",
            "sensor_missing_fraction",
            "sensor_missing_run_log1p",
            "hour_sin",
            "hour_cos",
            "temporal_hour",
            "LogCn2_diff_1",
            "LogCn2_diff_3",
        ]
        selected = [self.feature_names.index(name) for name in preferred if name in self.feature_names]
        if not selected:
            selected = list(range(min(input_size, 6)))
        self.selected_idx = selected
        physics_size = len(self.selected_idx) + 3

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.cells = nn.ModuleList(
            [RTDMCell(hidden_size, hidden_size, regime_size, physics_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, physics_size),
        )
        if attention_pool:
            self.temporal_score = nn.Sequential(
                nn.Linear(hidden_size + physics_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )
            self.context_gate = nn.Sequential(
                nn.Linear(hidden_size * 2 + physics_size, hidden_size),
                nn.Sigmoid(),
            )
        else:
            self.temporal_score = None
            self.context_gate = None
        self.head = nn.Sequential(
            nn.Linear(hidden_size + regime_size + physics_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )
        if stable_branch:
            self.stable_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.stable_head = nn.Linear(hidden_size, pred_len)
            self.output_gate = nn.Sequential(
                nn.Linear(hidden_size * 2 + regime_size + physics_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, pred_len),
                nn.Sigmoid(),
            )
        else:
            self.stable_lstm = None
            self.stable_head = None
            self.output_gate = None
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size + regime_size + physics_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size + regime_size + physics_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def _physics_summary(self, x):
        subset = x[..., self.selected_idx]
        current = subset
        mean = subset.mean(dim=1, keepdim=True).expand_as(current)
        delta = current - mean
        feat = torch.cat([current, delta[..., :1], subset.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True).expand(x.size(0), x.size(1), 1), mean[..., :1]], dim=-1)
        return self.physics_mlp(feat)

    def _forward_internal(self, x):
        z = self.input_proj(x)
        phys = self._physics_summary(x)
        batch_size, seq_len, _ = z.shape
        slow_states = [z.new_zeros(batch_size, self.hidden_size) for _ in self.cells]
        fast_states = [z.new_zeros(batch_size, self.hidden_size) for _ in self.cells]
        mixed_states = [z.new_zeros(batch_size, self.hidden_size) for _ in self.cells]
        regime_last = z.new_zeros(batch_size, self.cells[0].regime_size)
        regime_history = []
        top_history = []

        for t in range(seq_len):
            layer_input = z[:, t, :]
            phys_t = phys[:, t, :]
            for idx, cell in enumerate(self.cells):
                slow, fast, mixed, regime = cell(
                    layer_input,
                    slow_states[idx],
                    fast_states[idx],
                    mixed_states[idx],
                    phys_t,
                )
                slow_states[idx] = slow
                fast_states[idx] = fast
                mixed_states[idx] = mixed
                regime_last = regime
                layer_input = mixed
            regime_history.append(regime.unsqueeze(1))
            top_history.append(mixed_states[-1].unsqueeze(1))

        final_memory = mixed_states[-1]
        if self.attention_pool and top_history:
            memory_seq = torch.cat(top_history, dim=1)
            score_in = torch.cat([memory_seq, phys], dim=-1)
            attn = torch.softmax(self.temporal_score(score_in), dim=1)
            attended = (attn * memory_seq).sum(dim=1)
            gate = self.context_gate(torch.cat([final_memory, attended, phys[:, -1, :]], dim=-1))
            final_memory = gate * final_memory + (1.0 - gate) * attended

        final = torch.cat([final_memory, regime_last, phys[:, -1, :]], dim=-1)
        pred = self.head(final)
        if self.stable_branch:
            stable_out, _ = self.stable_lstm(x)
            stable_last = stable_out[:, -1, :]
            stable_pred = self.stable_head(stable_last)
            gate = self.output_gate(torch.cat([final_memory, stable_last, regime_last, phys[:, -1, :]], dim=-1))
            pred = gate * pred + (1.0 - gate) * stable_pred
        regime_seq = torch.cat(regime_history, dim=1) if regime_history else None
        return pred, final, regime_seq

    def forward(self, x):
        pred, _, _ = self._forward_internal(x)
        return pred

    def compute_loss(self, x, y, loss_cfg=None):
        loss_cfg = loss_cfg or {}
        pred, final, regime_seq = self._forward_internal(x)

        hard_weight = float(loss_cfg.get("hard_sample_weight", 0.5))
        risk_lambda = float(loss_cfg.get("risk_lambda", 0.2))
        volatility_lambda = float(loss_cfg.get("volatility_lambda", 0.1))
        regime_lambda = float(loss_cfg.get("regime_smooth_lambda", 0.02))
        risk_threshold = float(loss_cfg.get("risk_threshold", 0.8))

        sample_score = y.detach().abs().amax(dim=1, keepdim=True)
        sample_score = sample_score / (sample_score.mean() + 1e-6)
        sample_weight = 1.0 + hard_weight * sample_score
        point_loss = (((pred - y) ** 2) * sample_weight).mean()

        volatility_target = y.detach().abs().mean(dim=1)
        volatility_pred = self.volatility_head(final).squeeze(-1)
        volatility_loss = ((volatility_pred - volatility_target) ** 2).mean()

        risk_target = (y.detach().abs().amax(dim=1) > risk_threshold).float()
        risk_logit = self.risk_head(final).squeeze(-1)
        risk_loss = nn.functional.binary_cross_entropy_with_logits(risk_logit, risk_target)

        regime_smooth_loss = pred.new_tensor(0.0)
        if regime_seq is not None and regime_seq.size(1) > 1:
            regime_smooth_loss = (regime_seq[:, 1:, :] - regime_seq[:, :-1, :]).abs().mean()

        total = point_loss
        total = total + volatility_lambda * volatility_loss
        total = total + risk_lambda * risk_loss
        total = total + regime_lambda * regime_smooth_loss
        stats = {
            "point_loss": float(point_loss.detach().cpu()),
            "volatility_loss": float(volatility_loss.detach().cpu()),
            "risk_loss": float(risk_loss.detach().cpu()),
            "regime_smooth_loss": float(regime_smooth_loss.detach().cpu()),
        }
        return pred, total, stats
