import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR     = Path("models/onnx")   # matches model_loader.py
EMBEDDING_DIM = 384                   # DINOv2-small CLS token size
EPOCHS        = 30
BATCH_SIZE    = 16
LEARNING_RATE = 0.001


# ── Dataset ────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Expects data/sequences.json — a list of sequence objects:
      [{"images": ["path/a.jpg", "path/b.jpg", "path/c.jpg"]}, ...]

    Each consecutive pair (i, i+1) becomes one training sample:
      pos = earlier image (should score higher for sequence fitness)
      neg = later  image
    """
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self._pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for seq_obj in raw:
            imgs = seq_obj.get("images", [])
            for i in range(len(imgs) - 1):
                pos_emb = self._load_embedding(imgs[i])
                neg_emb = self._load_embedding(imgs[i + 1])
                self._pairs.append((pos_emb, neg_emb))

        print(f"Loaded {len(self._pairs)} training pairs from {len(raw)} sequences.")

    def _load_embedding(self, img_path: str) -> np.ndarray:
        # Try a pre-saved .npy sidecar file (any extension)
        npy_path = Path(img_path).with_suffix(".npy")
        if npy_path.exists():
            return np.load(str(npy_path)).astype(np.float32)
        # Fallback: random vector so the script runs even without real embeddings.
        # Replace this branch with your ONNX inference call if you want real training.
        return np.random.rand(EMBEDDING_DIM).astype(np.float32)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos, neg = self._pairs[idx]
        return torch.from_numpy(pos), torch.from_numpy(neg)


# ── Model ──────────────────────────────────────────────────────────────────────
class SequenceScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    data_path = Path("data/sequences.json")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Create it first.")
        return

    dataset = SequenceDataset(str(data_path))
    if len(dataset) == 0:
        print("ERROR: No training pairs found in sequences.json")
        return

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model     = SequenceScorer(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=0.2)  # wants score(pos) > score(neg)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for pos_emb, neg_emb in loader:
            # pos_emb, neg_emb: (batch, EMBEDDING_DIM) tensors from collate_fn
            target = torch.ones(pos_emb.size(0))  # 1 = pos should rank above neg

            optimizer.zero_grad()
            score_pos = model(pos_emb).squeeze(1)
            score_neg = model(neg_emb).squeeze(1)
            loss      = criterion(score_pos, score_neg, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch + 1:>3}/{EPOCHS}  loss={avg:.4f}")

    # ── Export to ONNX ─────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path   = MODEL_DIR / "sequence_scorer.onnx"
    dummy_input = torch.randn(1, EMBEDDING_DIM)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["embedding"],
        output_names=["score"],
        dynamic_axes={"embedding": {0: "batch_size"}, "score": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"Saved: {onnx_path}")


if __name__ == "__main__":
    train()
