import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

def compute_lags(ts, top_k=5, max_lag=24):
    """
    ts: time series of type np.ndarray, shape [seq_len, channels] <- same shape being used by the authors
    """
    x = np.asarray(ts, dtype=np.float32)

    # reduce multivariate TS to one signal for lag estimation
    if x.ndim == 2:
        x = x.mean(axis=1)

    x = x - x.mean()

    if np.allclose(x.std(), 0):
        return [0] * top_k

    max_lag = min(max_lag, len(x) - 1)
    scores = []

    for lag in range(1, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]

        if a.std() == 0 or b.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(a, b)[0, 1]

        scores.append((lag, abs(corr)))

    scores = sorted(scores, key=lambda z: z[1], reverse=True)
    return [lag for lag, _ in scores[:top_k]]


'''
    Same Template that has been used in the paper: https://arxiv.org/pdf/2410.16489
    Covers the following statistics:
        - min value
        - max value
        - median value
        - Top 5 Lags
'''

def build_ts_description(ts, task_description="This is a time series imputation task"):

    x = np.asarray(ts, dtype=np.float32)

    # As we are computationally constrained, I limited the amount of values (ts) to 384
    flat = x.flatten()
    max_ts_values = 64
    shown = flat[:max_ts_values]

    ts_content = ", ".join([f"{v:.4f}" for v in shown])

    if flat.size > max_ts_values:
        ts_content += ", ..."

    lags = compute_lags(x, top_k=5)

    text = (
        f"{task_description}. "
        f"The content is: {ts_content}. "
        f"Input statistics: "
        f"min value {np.min(x):.6f}, "
        f"max value {np.max(x):.6f}, "
        f"median value {np.median(x):.6f}, "
        f"top 5 lags {lags}."
    )

    return text


def build_feature_data(
    data_x,
    seq_len,
    save_path,
    device,
    model_name="gpt2",
    max_n=10000,
    batch_size=16,
):
    """
    Creates feature_data.pt for LLM-TS Integrator using GPT-2 instead of Llama 3B
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_windows = min(len(data_x) - seq_len + 1, max_n)

    texts = []
    for i in range(n_windows):
        seq_x = data_x[i : i + seq_len]
        text = build_ts_description(seq_x)
        texts.append(text)

    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = hidden * attention_mask
            emb = hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        all_embs.append(emb.cpu())

    feature_data = torch.cat(all_embs, dim=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feature_data, save_path)

    print("Saved Feature Data:", save_path)
    print("Shape of our Feature Data:", feature_data.shape)

    return feature_data