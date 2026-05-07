from src.models.deep_models import (
    CNNLSTMModel,
    GRUModel,
    LSTMModel,
    MLPModel,
    PatchTSTModel,
    RTDMNetModel,
    TCNModel,
    TFTModel,
    TransformerModel,
    VanillaLSTMModel,
)


def build_model(model_name: str, input_size: int, cfg: dict, dataset_meta: dict | None = None):
    seq_len = cfg["data"]["seq_len"]
    mcfg = cfg["models"]
    dataset_meta = dataset_meta or {}
    pred_len = int(dataset_meta.get("pred_len", cfg["data"]["pred_len"]))
    if model_name == "mlp":
        p = mcfg["mlp"]
        return MLPModel(
            input_size,
            seq_len=seq_len,
            hidden_sizes=p.get("hidden_sizes", [256, 128]),
            dropout=p.get("dropout", 0.2),
            pred_len=pred_len,
        )
    if model_name == "vanilla_lstm":
        p = mcfg["lstm"]
        return VanillaLSTMModel(
            input_size=input_size,
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            pred_len=pred_len,
            bidirectional=p.get("bidirectional", False),
        )
    if model_name == "lstm":
        p = mcfg["lstm"]
        return LSTMModel(
            input_size=input_size,
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            pred_len=pred_len,
            bidirectional=p.get("bidirectional", False),
            attention_pool=p.get("attention_pool", False),
            input_proj_size=p.get("input_proj_size"),
            ar_window=p.get("ar_window", 0),
            context_window=p.get("context_window", 12),
            feature_names=dataset_meta.get("feature_cols"),
        )
    if model_name == "gru":
        p = mcfg["gru"]
        return GRUModel(input_size, p["hidden_size"], p["num_layers"], p["dropout"], pred_len)
    if model_name == "transformer":
        p = mcfg["transformer"]
        return TransformerModel(
            input_size,
            p["d_model"],
            p["nhead"],
            p["num_layers"],
            p["ff_dim"],
            p["dropout"],
            pred_len,
            use_positional_encoding=p.get("use_positional_encoding", True),
            max_len=max(seq_len, p.get("max_len", 512)),
            pooling=p.get("pooling", "last"),
        )
    if model_name == "patchtst":
        p = mcfg["patchtst"]
        return PatchTSTModel(
            input_size=input_size,
            patch_len=p.get("patch_len", 6),
            stride=p.get("stride", 3),
            d_model=p.get("d_model", 96),
            nhead=p.get("nhead", 4),
            num_layers=p.get("num_layers", 3),
            ff_dim=p.get("ff_dim", 192),
            dropout=p.get("dropout", 0.1),
            pred_len=pred_len,
            pooling=p.get("pooling", "mean"),
            max_patches=p.get("max_patches", 128),
        )
    if model_name == "tft":
        p = mcfg["tft"]
        return TFTModel(
            input_size=input_size,
            hidden_size=p.get("hidden_size", 96),
            lstm_layers=p.get("lstm_layers", 1),
            num_heads=p.get("num_heads", 4),
            dropout=p.get("dropout", 0.1),
            pred_len=pred_len,
        )
    if model_name == "tcn":
        p = mcfg["tcn"]
        return TCNModel(input_size, p["channels"], p["kernel_size"], p["dropout"], pred_len)
    if model_name == "cnn_lstm":
        p = mcfg["cnn_lstm"]
        return CNNLSTMModel(
            input_size,
            p["cnn_channels"],
            p["lstm_hidden"],
            p["lstm_layers"],
            p["dropout"],
            pred_len,
        )
    if model_name == "rtdm":
        p = mcfg["rtdm"]
        return RTDMNetModel(
            input_size=input_size,
            hidden_size=p["hidden_size"],
            regime_size=p.get("regime_size", 4),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.1),
            pred_len=pred_len,
            feature_names=dataset_meta.get("feature_cols"),
            stable_branch=p.get("stable_branch", False),
            attention_pool=p.get("attention_pool", False),
        )
    raise ValueError(f"Unsupported model: {model_name}")
