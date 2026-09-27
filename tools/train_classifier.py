"""Train the beacon-identification CNN and export it for the application.

Input : the dataset from tools/generate_dataset.py (data/beacon_dataset.npz)
Output: models/beacon_cnn.onnx        model used by the app (via OpenCV DNN)
        models/beacon_cnn.pt          PyTorch weights (for further training)
        models/training_report.json   metrics: CNN vs classical blink detector
        models/training_curves.png    loss / accuracy per epoch
        models/confusion_matrix.png   validation confusion matrices

Network ("BeaconNet"): the candidate's last T patches are stacked as T input
channels, so the first convolution learns spatio-temporal filters (blink
pattern + spot shape + motion consistency) in one step.

    Conv3x3(T->32)+BN+ReLU -> MaxPool -> Conv3x3(32->64)+BN+ReLU -> MaxPool
    -> Conv3x3(64->64)+BN+ReLU -> GlobalAvgPool -> Dropout -> Linear(64->2)

    python tools/train_classifier.py
    python tools/train_classifier.py --epochs 40
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsoc.identification import BlinkScorer, aperture_flux, normalize_stack  # noqa: E402

KIND_NAMES = ["beacon", "steady_decoy", "wrong_rate_decoy", "near_rate_decoy", "glint", "other",
              "harmonic_decoy"]


class BeaconNet(nn.Module):
    def __init__(self, frames):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(frames, 32), nn.MaxPool2d(2),
            *block(32, 64), nn.MaxPool2d(2),
            *block(64, 64), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.features(x))


def augment(x):
    """Random flips / 90-degree rotations (the problem is rotation-invariant)
    plus a little extra noise. x: (B, T, S, S) float tensor."""
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
    return x + 0.02 * torch.randn_like(x)


def binary_metrics(y_true, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, p)) if len(set(y_true)) > 1 else float("nan")
    return {"accuracy": round((tp + tn) / len(y_true), 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "roc_auc": round(auc, 4), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def per_kind_accuracy(kind, y_true, p):
    out = {}
    for k, name in enumerate(KIND_NAMES):
        m = kind == k
        if m.any():
            out[name] = {"n": int(m.sum()), "accuracy": round(float(((p[m] >= 0.5) == y_true[m]).mean()), 4)}
    return out


def blink_baseline(X, beacon_hz, fps):
    """Score every sample with the classical blink detector."""
    scorer = BlinkScorer(beacon_hz, fps)

    class _T:                                   # minimal tracklet stand-in
        def __init__(self, i, flux):
            self.id, self.flux = i, flux

    items = [_T(i, [aperture_flux(p) for p in s]) for i, s in enumerate(X)]
    scores = scorer.score(items)
    return np.array([scores[i] for i in range(len(X))])


def predict(model, X, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(X[i:i + batch]), 1)[:, 1])
    return torch.cat(out).numpy()


def save_plots(history, cms, path_curves, path_cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, history["train_loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, history["train_acc"], label="train")
    ax[1].plot(ep, history["val_acc"], label="validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path_curves, dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.8))
    for ax, (title, c) in zip(np.atleast_1d(axes), cms.items()):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["not beacon", "beacon"]); ax.set_yticks([0, 1], ["not beacon", "beacon"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path_cm, dpi=130); plt.close(fig)


def export_onnx(model, frames, patch, path):
    model.eval()
    dummy = torch.zeros(1, frames, patch, patch)
    # The TorchScript exporter imports the separate `onnx` package only to
    # embed custom ONNX-script functions, which this plain CNN does not use.
    # Skipping that step makes export work even when `onnx` is missing or
    # broken (e.g. version clash with `ml_dtypes`).
    from torch.onnx._internal import onnx_proto_utils
    original = onnx_proto_utils._add_onnxscript_fn
    onnx_proto_utils._add_onnxscript_fn = lambda proto, *args, **kwargs: proto
    try:
        torch.onnx.export(model, dummy, str(path), input_names=["clip"], output_names=["logits"],
                          dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=13, dynamo=False)
    finally:
        onnx_proto_utils._add_onnxscript_fn = original


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "beacon_dataset_v2.npz"))
    # Default output is a CANDIDATE folder so a retrain never overwrites the model
    # the application uses (models/beacon_cnn.onnx); promote it deliberately.
    p.add_argument("--out", default=str(ROOT / "models" / "candidate"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; evaluate and export the existing models/beacon_cnn.pt")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    X_raw, y, kind, split = d["X"], d["y"].astype(np.int64), d["kind"], d["split"]
    beacon_hz = float(d["beacon_hz"])
    n, frames, patch, _ = X_raw.shape
    print(f"Dataset: {n} samples, clips of {frames} x {patch}x{patch}, beacon code {beacon_hz} Hz")

    X = torch.from_numpy(np.stack([normalize_stack(s) for s in X_raw]))
    Y = torch.from_numpy(y)
    tr, va = split == 0, split == 1
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    print(f"Train {len(Ytr)} ({int(Ytr.sum())} beacon) | validation {len(Yva)} ({int(Yva.sum())} beacon)")

    model = BeaconNet(frames)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"BeaconNet: {n_params:,} parameters")
    pos = float(Ytr.float().mean())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1 / (1 - pos), 1 / pos]) / 2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs *
                                                ((len(Ytr) + args.batch - 1) // args.batch))

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_state, best_val = None, float("inf")
    out = Path(args.out)
    if args.eval_only:
        best_state = torch.load(out / "beacon_cnn.pt")
        history_file = out / "training_history.json"
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding="utf-8"))
    t0 = time.time()
    for epoch in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Ytr))
        tot_loss = tot_ok = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = augment(Xtr[idx]), Ytr[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_ok += float((logits.argmax(1) == yb).sum())
        model.eval()
        with torch.no_grad():
            lv = model(Xva)
            val_loss = float(loss_fn(lv, Yva))
            val_acc = float((lv.argmax(1) == Yva).float().mean())
        history["train_loss"].append(tot_loss / len(Ytr)); history["train_acc"].append(tot_ok / len(Ytr))
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.3f}"
              f"  |  val loss {val_loss:.4f} acc {val_acc:.3f}   [{time.time() - t0:.0f} s]", flush=True)

    model.load_state_dict(best_state)

    # --- Evaluation on the held-out validation episodes -------------------
    y_va = Yva.numpy()
    p_cnn = predict(model, Xva)
    p_blink = blink_baseline(X_raw[va], beacon_hz, args.fps)
    t1 = time.perf_counter()
    predict(model, Xva[:256])
    cnn_ms = (time.perf_counter() - t1) / min(256, len(Xva)) * 1e3
    report = {
        "dataset": {"samples": int(n), "train": int(tr.sum()), "validation": int(va.sum()),
                    "clip_frames": int(frames), "patch_px": int(patch), "beacon_hz": beacon_hz},
        "model": {"name": "BeaconNet", "parameters": int(n_params), "epochs": args.epochs,
                  "cpu_ms_per_candidate_torch": round(cnn_ms, 3)},
        "cnn": binary_metrics(y_va, p_cnn),
        "blink_baseline": binary_metrics(y_va, p_blink),
        "cnn_per_kind": per_kind_accuracy(kind[va], y_va, p_cnn),
        "blink_per_kind": per_kind_accuracy(kind[va], y_va, p_blink),
    }

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "beacon_cnn.pt")
    if history["train_loss"]:
        (out / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    export_onnx(model, frames, patch, out / "beacon_cnn.onnx")

    # Check that OpenCV's DNN (used by the app) reproduces PyTorch exactly.
    import cv2
    net = cv2.dnn.readNetFromONNX(str(out / "beacon_cnn.onnx"))
    sample = Xva[:32].numpy()
    net.setInput(sample)
    cv_logits = net.forward()
    with torch.no_grad():
        pt_logits = model(Xva[:32]).numpy()
    report["onnx_max_abs_diff"] = float(np.abs(cv_logits - pt_logits).max())

    with open(out / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    save_plots(history, {"CNN (BeaconNet)": report["cnn"]["confusion"],
                         "Classical blink detector": report["blink_baseline"]["confusion"]},
               out / "training_curves.png", out / "confusion_matrix.png")

    print("\nValidation results (scenes never seen in training):")
    print(f"{'metric':12s} {'CNN':>8s} {'blink':>8s}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:12s} {report['cnn'][k]:8.3f} {report['blink_baseline'][k]:8.3f}")
    print("\nAccuracy per candidate type:")
    for name in report["cnn_per_kind"]:
        c, b = report["cnn_per_kind"][name], report["blink_per_kind"][name]
        print(f"  {name:18s} n={c['n']:5d}   CNN {c['accuracy']:.3f}   blink {b['accuracy']:.3f}")
    print(f"\nONNX (OpenCV DNN) vs PyTorch max difference: {report['onnx_max_abs_diff']:.2e}")
    print(f"Saved model and report to {out}")


if __name__ == "__main__":
    main()
