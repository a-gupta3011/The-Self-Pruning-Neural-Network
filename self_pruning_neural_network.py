from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import certifi
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# SSL robustness for CIFAR-10 download on macOS / Python installations.
# -----------------------------------------------------------------------------
try:
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """Linear layer with a learnable gate per weight.

    gate_scores -> sigmoid(gate_scores / temperature) = gate values in (0, 1)
    pruned_weight = weight * gates
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gate_init: float = -0.25):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters(gate_init=gate_init)

    def reset_parameters(self, gate_init: float = -0.25) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, gate_init)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def gates(self, temperature: float = 1.0) -> torch.Tensor:
        temperature = max(float(temperature), 1e-3)
        return torch.sigmoid(self.gate_scores / temperature)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        g = self.gates(temperature)
        pruned_weight = self.weight * g
        return F.linear(x, pruned_weight, self.bias)

    def sparsity_penalty(self, temperature: float = 1.0) -> torch.Tensor:
        # Mean gate value, so lambda is easy to tune.
        return self.gates(temperature).mean()

    @torch.no_grad()
    def gate_values(self, temperature: float = 1.0) -> torch.Tensor:
        return self.gates(temperature).detach().flatten()


class PrunableMLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 768)
        self.fc2 = PrunableLinear(768, 256)
        self.fc3 = PrunableLinear(256, num_classes)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x, temperature))
        x = self.dropout(x)
        x = F.relu(self.fc2(x, temperature))
        x = self.dropout(x)
        x = self.fc3(x, temperature)
        return x

    def sparsity_penalty(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.stack([
            self.fc1.sparsity_penalty(temperature),
            self.fc2.sparsity_penalty(temperature),
            self.fc3.sparsity_penalty(temperature),
        ]).mean()

    @torch.no_grad()
    def all_gate_values(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.cat([
            self.fc1.gate_values(temperature),
            self.fc2.gate_values(temperature),
            self.fc3.gate_values(temperature),
        ])

    @torch.no_grad()
    def sparsity_percent(self, threshold: float = 1e-2, temperature: float = 0.35) -> float:
        gates = self.all_gate_values(temperature)
        return (gates < threshold).float().mean().item() * 100.0


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    batch_size: int,
    quick: bool,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    if quick:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(train_set), size=min(10000, len(train_set)), replace=False)
        test_idx = rng.choice(len(test_set), size=min(2000, len(test_set)), replace=False)
        train_set = Subset(train_set, train_idx.tolist())
        test_set = Subset(test_set, test_idx.tolist())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------------------------

@dataclass
class EpochStats:
    loss: float
    acc: float


@torch.no_grad()
def evaluate(model: PrunableMLP, loader: DataLoader, device: torch.device, temperature: float) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, temperature=temperature)
        loss = F.cross_entropy(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs
    return EpochStats(loss=total_loss / total, acc=total_correct / total)


def temperature_schedule(epoch: int, total_epochs: int, start_temp: float = 2.0, end_temp: float = 0.35) -> float:
    if total_epochs <= 1:
        return end_temp
    frac = epoch / max(total_epochs - 1, 1)
    return start_temp + frac * (end_temp - start_temp)


def sparsity_schedule(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    denom = max(total_epochs - warmup_epochs - 1, 1)
    return (epoch - warmup_epochs) / denom


def train_one_lambda(
    lambda_value: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    gate_lr: float,
    weight_decay: float,
    threshold: float,
) -> Tuple[PrunableMLP, dict]:
    model = PrunableMLP().to(device)

    weight_params = []
    gate_params = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": weight_params, "lr": lr, "weight_decay": weight_decay},
            {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0},
        ]
    )

    history = {
        "lambda": lambda_value,
        "epochs": [],
    }

    warmup_epochs = max(1, epochs // 4)

    print("=" * 80)
    print(f"Training model with lambda = {lambda_value:.2e}")
    print("=" * 80)

    best_state = None
    best_test_acc = -1.0

    for epoch in range(epochs):
        model.train()
        temp = temperature_schedule(epoch, epochs)
        sparsity_scale = sparsity_schedule(epoch, epochs, warmup_epochs)

        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, temperature=temp)
            cls_loss = F.cross_entropy(logits, y)
            gate_loss = model.sparsity_penalty(temperature=temp)
            loss = cls_loss + (lambda_value * sparsity_scale * gate_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += bs

        train_loss = total_loss / total
        train_acc = total_correct / total
        test_stats = evaluate(model, test_loader, device, temperature=temp)
        sparsity = model.sparsity_percent(threshold=threshold, temperature=temp)

        print(
            f"[λ={lambda_value:.2e}] Epoch {epoch+1:02d}/{epochs} | "
            f"train loss={train_loss:.4f} | train acc={train_acc:.4f} | "
            f"test acc={test_stats.acc:.4f} | sparsity={sparsity:.2f}% | temp={temp:.2f}"
        )

        history["epochs"].append(
            {
                "epoch": epoch + 1,
                "temperature": temp,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_stats.loss,
                "test_acc": test_stats.acc,
                "sparsity_percent": sparsity,
            }
        )

        if test_stats.acc > best_test_acc:
            best_test_acc = test_stats.acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_temp = 0.25
    final_test = evaluate(model, test_loader, device, temperature=final_temp)
    final_sparsity = model.sparsity_percent(threshold=threshold, temperature=final_temp)

    history["final_test_acc"] = final_test.acc
    history["final_test_loss"] = final_test.loss
    history["final_sparsity_percent"] = final_sparsity

    return model, history


# -----------------------------------------------------------------------------
# Results / plotting
# -----------------------------------------------------------------------------

def save_results_csv(results: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "test_accuracy", "sparsity_percent"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "lambda": row["lambda"],
                    "test_accuracy": row["test_accuracy"],
                    "sparsity_percent": row["sparsity_percent"],
                }
            )


def save_summary_json(results: Sequence[dict], best_row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "best_model": best_row,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_gate_distribution(model: PrunableMLP, output_path: Path, temperature: float = 0.25) -> None:
    gates = model.all_gate_values(temperature=temperature).cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(gates, bins=60)
    plt.title("Distribution of Final Gate Values (Best Model)")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-pruning neural network on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use smaller subsets for a fast sanity run")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 2.0],
        help="Lambda values to compare",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        quick=args.quick,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_model = None
    best_row = None
    best_acc = -1.0

    for lam in args.lambdas:
        model, history = train_one_lambda(
            lambda_value=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            gate_lr=args.gate_lr,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
        )

        row = {
            "lambda": lam,
            "test_accuracy": history["final_test_acc"],
            "sparsity_percent": history["final_sparsity_percent"],
        }
        all_results.append(row)

        if row["test_accuracy"] > best_acc:
            best_acc = row["test_accuracy"]
            best_model = model
            best_row = row

    print("\nFinal comparison:")
    for row in all_results:
        print(f"λ={row['lambda']:.2e} | test acc={row['test_accuracy']:.4f} | sparsity={row['sparsity_percent']:.2f}%")

    print(
        f"\nBest model selected: λ={best_row['lambda']:.2e} "
        f"(test acc={best_row['test_accuracy']:.4f}, sparsity={best_row['sparsity_percent']:.2f}%)"
    )

    save_results_csv(all_results, out_dir / "results.csv")
    save_summary_json(all_results, best_row, out_dir / "summary.json")
    plot_gate_distribution(best_model, out_dir / "gate_distribution_best.png", temperature=0.25)

    print(f"Saved results to: {out_dir / 'results.csv'}")
    print(f"Saved gate histogram to: {out_dir / 'gate_distribution_best.png'}")
    print(f"Saved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
