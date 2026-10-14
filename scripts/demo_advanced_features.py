"""Reproducible college-demo for DANN, temporal gating, and dynamic quantization.

By default this script uses generated synthetic data so it runs without private or
large datasets. Synthetic results are software demonstrations, not river-science
results. A future real-data adapter can replace make_demo_data() while preserving
these model/evaluation interfaces.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Function
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "advanced_demo_001"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_demo_data(n_source: int = 480, n_target: int = 240, seed: int = 42):
    """Create labelled source-domain and shifted target-domain multimodal samples."""
    rng = np.random.default_rng(seed)

    def sample(n: int, target_domain: bool):
        sat = rng.normal(0.0, 1.0, (n, 10)).astype(np.float32)
        ground = rng.normal(0.0, 1.0, (n, 3)).astype(np.float32)
        # Deliberate domain shift: spectral response and ground measurement scales differ.
        if target_domain:
            sat = sat * np.array([1.1, .9, 1.2, .8, 1.1, .9, 1.15, .85, 1.2, .95], np.float32) + .65
            ground = ground * np.array([1.2, .85, 1.1], np.float32) + .45
        latent = (0.38 * sat[:, 2] - 0.22 * sat[:, 7] +
                  0.48 * ground[:, 0] - 0.25 * ground[:, 1] +
                  0.12 * sat[:, 4] * ground[:, 0])
        # Same underlying relationship; target measurements include modest noise.
        y = (3.0 + 0.8 * latent + rng.normal(0, .18 if not target_domain else .22, n)).astype(np.float32)
        # Time-offset features represent the age of the latest satellite and ground observation.
        age = rng.uniform(0, 5, (n, 2)).astype(np.float32)
        if target_domain:
            age[:, 0] = np.minimum(age[:, 0] + 1.5, 7.0)
        return sat, ground, age, y

    return sample(n_source, False), sample(n_target, True)


def metrics(y, pred):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None}


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class DannModel(nn.Module):
    def __init__(self, input_dim: int = 13, hidden: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 12), nn.ReLU())
        self.regressor = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1))
        self.domain_classifier = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x, grl_strength: float = 0.0):
        z = self.encoder(x)
        prediction = self.regressor(z).squeeze(-1)
        domain_logits = self.domain_classifier(GradientReverse.apply(z, grl_strength))
        return prediction, domain_logits


class TemporalGateModel(nn.Module):
    """Two-branch MLP whose learned gate also sees observation-age features."""
    def __init__(self):
        super().__init__()
        self.satellite_branch = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.ground_branch = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(18, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, sat, ground, ages):
        s = self.satellite_branch(sat)
        g = self.ground_branch(ground)
        gate = self.gate(torch.cat([s, g, ages], dim=1))
        fused = gate * s + (1.0 - gate) * g
        return self.head(fused).squeeze(-1), gate.squeeze(-1)


class PlainFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1))

    def forward(self, sat, ground, ages):
        return self.net(torch.cat([sat, ground, ages], dim=1)).squeeze(-1)


def as_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train_dann(source, target, epochs=220):
    ss, sg, sa, sy = source
    ts, tg, ta, ty = target
    source_x = np.concatenate([ss, sg], axis=1)
    target_x = np.concatenate([ts, tg], axis=1)
    scaler = StandardScaler().fit(source_x)
    sx = as_tensor(scaler.transform(source_x))
    tx = as_tensor(scaler.transform(target_x))
    sy_t, ty_t = as_tensor(sy), as_tensor(ty)
    # Labelled source-domain-only baseline.
    baseline = DannModel()
    opt = torch.optim.Adam(baseline.parameters(), lr=0.008, weight_decay=1e-4)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = baseline(sx)
        loss = mse(pred, sy_t)
        loss.backward(); opt.step()
    baseline.eval()
    with torch.no_grad():
        baseline_pred, _ = baseline(tx)
    # DANN: source labels train the regressor; source+target domain labels train domain head.
    model = DannModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    domain_labels = torch.cat([torch.zeros(len(sx), dtype=torch.long), torch.ones(len(tx), dtype=torch.long)])
    combined = torch.cat([sx, tx])
    for epoch in range(epochs):
        strength = min(1.0, epoch / max(1, epochs * .35)) * .35
        opt.zero_grad()
        pred_s, _ = model(sx, strength)
        _, domain_logits = model(combined, strength)
        loss = mse(pred_s, sy_t) + 0.18 * nn.CrossEntropyLoss()(domain_logits, domain_labels)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        dann_pred, domain_logits = model(tx, 0.0)
    rows = [
        {"method": "source_only_mlp", **metrics(ty, baseline_pred.numpy())},
        {"method": "dann_demo", **metrics(ty, dann_pred.numpy())},
    ]
    return rows, {"model": model, "baseline": baseline, "domain_accuracy": float((domain_logits.argmax(1) == 1).float().mean())}


def train_temporal(source, target, epochs=260):
    ss, sg, sa, sy = source
    ts, tg, ta, ty = target
    # Train on source domain; evaluate on shifted target domain.
    scaler_sat = StandardScaler().fit(ss)
    scaler_ground = StandardScaler().fit(sg)
    scaler_age = StandardScaler().fit(sa)
    scaler_y = StandardScaler().fit(sy.reshape(-1, 1))
    def pack(data):
        sat, ground, ages, y = data
        return (as_tensor(scaler_sat.transform(sat)), as_tensor(scaler_ground.transform(ground)),
                as_tensor(scaler_age.transform(ages)), as_tensor(scaler_y.transform(y.reshape(-1, 1)).ravel()))
    train = pack(source); test = pack(target)
    gate_model = TemporalGateModel()
    plain_model = PlainFusionModel()
    optimizers = [torch.optim.Adam(gate_model.parameters(), lr=.008), torch.optim.Adam(plain_model.parameters(), lr=.008)]
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizers[0].zero_grad(); pred, _ = gate_model(*train[:3]); loss_fn(pred, train[3]).backward(); optimizers[0].step()
        optimizers[1].zero_grad(); pred = plain_model(*train[:3]); loss_fn(pred, train[3]).backward(); optimizers[1].step()
    gate_model.eval(); plain_model.eval()
    with torch.no_grad():
        gate_scaled, gates = gate_model(*test[:3]); plain_scaled = plain_model(*test[:3])
        y_true = scaler_y.inverse_transform(test[3].numpy().reshape(-1, 1)).ravel()
        gate_pred = scaler_y.inverse_transform(gate_scaled.numpy().reshape(-1, 1)).ravel()
        plain_pred = scaler_y.inverse_transform(plain_scaled.numpy().reshape(-1, 1)).ravel()
    rows = [
        {"method": "ordinary_concat_fusion", **metrics(y_true, plain_pred)},
        {"method": "temporal_attention_gate", **metrics(y_true, gate_pred)},
    ]
    return rows, {"model": gate_model, "scalers": (scaler_sat, scaler_ground, scaler_age, scaler_y),
                  "gate_mean": float(gates.mean()), "gate_std": float(gates.std())}


def benchmark_quantization(model, example_inputs, output_dir: Path):
    model.eval()
    # Dynamic INT8 quantization converts Linear layers, a supported CPU inference path.
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    def timed(candidate, repeats=100):
        with torch.no_grad():
            candidate(*example_inputs)
            start = time.perf_counter()
            for _ in range(repeats):
                candidate(*example_inputs)
            return (time.perf_counter() - start) * 1000 / repeats
    fp_path, int_path = output_dir / "temporal_gate_fp32.pt", output_dir / "temporal_gate_int8.pt"
    torch.save(model.state_dict(), fp_path)
    torch.save(quantized.state_dict(), int_path)
    with torch.no_grad():
        fp = model(*example_inputs)[0]
        qout = quantized(*example_inputs)[0]
        if isinstance(fp, tuple): fp = fp[0]
        if isinstance(qout, tuple): qout = qout[0]
        max_delta = float(torch.max(torch.abs(fp - qout)).item())
    return {
        "method": "PyTorch dynamic INT8 quantization of Linear layers",
        "fp32_state_dict_bytes": int(fp_path.stat().st_size),
        "int8_state_dict_bytes": int(int_path.stat().st_size),
        "state_dict_size_reduction_percent": float((1 - int_path.stat().st_size / max(fp_path.stat().st_size, 1)) * 100),
        "fp32_latency_ms_per_batch": timed(model),
        "int8_latency_ms_per_batch": timed(quantized),
        "max_absolute_output_delta_standardized": max_delta,
        "note": "Benchmark is hardware and batch-size dependent; state-dict size is not full deployment package size."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for a quick smoke demo")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    source, target = make_demo_data(seed=args.seed)
    epochs = 70 if args.quick else 220
    dann_rows, dann_models = train_dann(source, target, epochs=epochs)
    temporal_rows, temporal = train_temporal(source, target, epochs=epochs + 40)
    sat, ground, ages, _ = target
    example_inputs = (as_tensor(sat[:32]), as_tensor(ground[:32]), as_tensor(ages[:32]))
    quant_report = benchmark_quantization(temporal["model"], example_inputs, args.output_dir)

    results = pd.DataFrame(dann_rows + temporal_rows)
    results.to_csv(args.output_dir / "model_comparison.csv", index=False)
    pd.DataFrame([quant_report]).to_csv(args.output_dir / "quantization_benchmark.csv", index=False)
    # Plot the target-domain MAE comparison.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(results["method"], results["mae"])
    ax.set_ylabel("Target-domain MAE (synthetic BOD units)")
    ax.set_title("College demo: cross-domain and temporal fusion comparisons")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)
    report = {
        "title": "Advanced fusion feature demo",
        "data_source": "synthetic generated demonstration data; not CPCB/USGS observations",
        "seed": args.seed,
        "source_samples": len(source[3]), "target_samples": len(target[3]),
        "dann_domain_classifier_target_accuracy": dann_models["domain_accuracy"],
        "temporal_gate_mean": temporal["gate_mean"], "temporal_gate_std": temporal["gate_std"],
        "comparisons": results.to_dict(orient="records"),
        "quantization": quant_report,
        "limitations": [
            "All metrics use synthetic demo data and must not be presented as real river-water-quality performance.",
            "DANN result is a pedagogical demonstration, not validated cross-country adaptation.",
            "Temporal gating is learned from synthetic observation-age features.",
            "Dynamic INT8 quantization applies to supported Linear layers; CPU timing varies by hardware.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Advanced-feature demo completed.")
    print("Data source: SYNTHETIC demonstration data (not CPCB/USGS observations).")
    print("\nModel comparison:")
    print(results.to_string(index=False))
    print("\nQuantization benchmark:")
    print(json.dumps(quant_report, indent=2))
    print("\nOutput directory:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
