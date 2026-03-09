from sklearn.model_selection import StratifiedKFold
from deepseekmoe import DeepseekMoE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from typing import List, Tuple, Dict, Union, Any
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # modify as needed
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from peft import C3AConfig, get_peft_model  # import as needed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, accuracy_score
import umap
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F


# -----------------------------
# Data
# -----------------------------
class SequenceDatasetHF(TorchDataset):
    def __init__(self, dataset: Any, col_name: str = 'seqs', label_col: str = 'labels', max_length: int = 2048):
        self.seqs = dataset[col_name].tolist()
        self.labels = dataset[label_col].tolist()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[str, Union[float, int]]:
        seq = self.seqs[idx][:self.max_length]
        label = self.labels[idx]
        return seq, label


class SequenceCollator:
    def __init__(self, tokenizer: Any, regression: bool = False):
        self.tokenizer = tokenizer
        self.regression = regression

    def __call__(self, batch: List[Tuple[str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs, labels = zip(*batch)
        labels = torch.tensor(labels)
        if self.regression:
            labels = labels.float()
        else:
            labels = labels.long()

        tokenized = self.tokenizer(
            seqs,
            padding='longest',
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

def build_peft_config(peft_method: str, target_modules: list):
    peft_method = peft_method.lower()

    if peft_method == "c3a":
        return C3AConfig(
            block_size=128,
            bias="none",
            target_modules=target_modules,
        )

    elif peft_method == "lora":
        return LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.01,
            bias="none",
            target_modules=target_modules,
        )

    elif peft_method == "miss":
        return MissConfig(
            r=8,
            miss_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )

    elif peft_method == "oft":
        return OFTConfig(
            oft_block_size=32,
            bias="none",
            target_modules=target_modules,
        )

    elif peft_method == "randlora":
        return RandLoraConfig(
            r=32,
            projection_prng_key=0,
            save_projection=True,
            sparse=False,
            very_sparse=False,
            randlora_dropout=0.0,
            randlora_alpha=128,
            fan_in_fan_out=False,
            target_modules=target_modules,
        )

    elif peft_method == "shira":
        return ShiraConfig(
            r=16,
            mask_type="random",
            random_seed=None,
            target_modules=target_modules,
            fan_in_fan_out=False,
            init_weights=True,
            modules_to_save=None,
        )

    else:
        raise ValueError(
            f"Unsupported peft_method: {peft_method}. "
            f"Choose from ['c3a', 'lora', 'miss', 'oft', 'randlora', 'shira']"
        )
# -----------------------------
# Model init
# -----------------------------
def initialize_model(model_name: str, num_labels: int, use_peft: bool = True, peft_method: str = 'c3a', peft_config: Any = None): #Modify peft_method as needed
    print(f"Loading model {model_name} with {num_labels} labels...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=num_labels
    )
    tokenizer = model.tokenizer

    # Replace the classifier head with DeepSeekMOE
    hidden_size = model.config.hidden_size * 2

    class MoEConfig:
        def __init__(self, hidden_size, num_labels):
            self.num_experts_per_tok = 2
            self.n_routed_experts = 3
            self.scoring_func = 'softmax'
            self.aux_loss_alpha = 0.3
            self.norm_topk_prob = True
            self.input_size = hidden_size
            self.output_dim = num_labels
            self.hidden_dim = 128

    moe_config = MoEConfig(hidden_size, num_labels)
    model.classifier = DeepseekMoE(moe_config)
    print("Replaced classifier head with DeepSeekMOE")

    if use_peft:
        target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
        if peft_config is None:
            peft_config = build_peft_config(peft_method, target_modules)

        model = get_peft_model(model, peft_config)
        print(f"Applied PEFT method: {peft_method}")


        # Unfreeze the MoE classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage trained: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer


def save_finetuned_weights(model, save_path: str):
    """Save only trainable parameters as tensors (safe for reload)."""
    state = model.state_dict()
    trainable = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in state:
            trainable[name] = state[name].detach().cpu()
    torch.save(trainable, save_path)
    print(f"Saved fine-tuned weights to {save_path} (N={len(trainable)})")


def load_finetuned_weights(model, load_path: str, device: torch.device):
    finetuned_weights = torch.load(load_path, map_location=device)
    model_state_dict = model.state_dict()
    missing = 0
    for name, tensor in finetuned_weights.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(tensor.to(model_state_dict[name].device))
        else:
            missing += 1
    model.load_state_dict(model_state_dict)
    if missing:
        print(f"Warning: {missing} params not found in model state_dict")
    print(f"Loaded fine-tuned weights from {load_path}")


def compute_metrics_classification(logits, labels):
    logits = logits[0] if isinstance(logits, tuple) else logits
    y_true = np.asarray(labels).astype(int)
    y_pred = np.argmax(logits, axis=1).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_prob = probs[:, 1]

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "sn": sn, "sp": sp, "mcc": mcc, "f1": f1, "auc": auc}


def print_metrics(metrics, prefix=""):
    print(f"{prefix}Metrics:")
    print(f"  Acc: {metrics['acc']:.4f}")
    print(f"  sn:  {metrics['sn']:.4f}")
    print(f"  sp:  {metrics['sp']:.4f}")
    print(f"  mcc: {metrics['mcc']:.4f}")
    print(f"  f1:  {metrics['f1']:.4f}")
    print(f"  auc: {metrics['auc']:.4f}")


def save_test_probabilities(
    logits: np.ndarray,
    labels: np.ndarray,
    out_csv: str,
    ids: np.ndarray = None
):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    p1 = probs[:, 1]
    pred = (p1 >= 0.5).astype(int)

    df = pd.DataFrame({
        "id": ids if ids is not None else np.arange(len(labels)),
        "y_true": labels.astype(int),
        "p_pos": p1.astype(float),
        "y_pred": pred
    })
    df.to_csv(out_csv, index=False)
    print(f"Saved test probabilities to {out_csv} (N={len(df)})")
    return df


# -----------------------------
# Density plot (overwrite each fold)
# -----------------------------
def plot_prediction_confidence_density(
    df_probs: pd.DataFrame,
    out_pdf: str,
    prob_col: str = "p_pos",
    pred_col: str = "y_pred",
    method_name: str = "PEFT"
):
    pos = df_probs[df_probs[pred_col] == 1][prob_col].values
    neg = df_probs[df_probs[pred_col] == 0][prob_col].values

    plt.figure(figsize=(8, 6))

    if len(pos) > 1:
        sns.kdeplot(
            pos,
            fill=True,
            alpha=0.35,
            linewidth=2,
            label="Predicted positive"
        )
    else:
        plt.hist(
            pos,
            bins=20,
            alpha=0.35,
            density=True,
            label="Predicted positive"
        )

    if len(neg) > 1:
        sns.kdeplot(
            neg,
            fill=True,
            alpha=0.35,
            linewidth=2,
            label="Predicted negative"
        )
    else:
        plt.hist(
            neg,
            bins=20,
            alpha=0.35,
            density=True,
            label="Predicted negative"
        )

    plt.title(f"Confidence distribution by {method_name}")
    plt.xlabel("Predicted probability P(y=1)")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined density plot to {out_pdf}")


def visualize_umap(model, data_loader, device, save_path="outputs/umap.pdf"):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="UMAP Embedding Extract"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)
            all_labels.append(labels)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"UMAP: collected {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=all_labels,
        cmap="coolwarm",
        alpha=0.7,
        s=20
    )
    plt.colorbar(scatter, label="Label")
    plt.title("UMAP Visualization")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"UMAP visualization saved to {save_path}")


# -----------------------------
# Temperature scaling
# -----------------------------
def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 2000) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.tensor(val_logits, dtype=torch.float32, device=device)
    labels = torch.tensor(val_labels, dtype=torch.long, device=device)

    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-3)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = torch.exp(log_T).detach().cpu().item()
    print(f"[TempScaling] Fitted temperature T = {T:.4f}")
    return float(T)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(T, 1e-6)


# -----------------------------
# Train / Eval (support grad accumulation)
# -----------------------------
def train_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps: int = 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss = loss / max(1, gradient_accumulation_steps)
        loss.backward()

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * max(1, gradient_accumulation_steps)
        progress_bar.set_postfix({'loss': float(loss.item() * max(1, gradient_accumulation_steps))})

    if (len(train_loader) % max(1, gradient_accumulation_steps)) != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / len(train_loader)


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return total_loss / len(data_loader), all_predictions, all_labels


def train_classification_model_10fold(
    model_name: str = 'ESMC',
    use_peft: bool = True,
    peft_method: str = "c3a",
    custom_peft_config: Any = None,
    batch_size: int = 8,
    learning_rate: float = 5e-4,  # Others: 5e-5
    num_epochs: int = 10,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    patience: int = 3,
    n_splits: int = 10,
    random_state: int = 42,
    temp_scale_multiplier: float = 2.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    def parse_fasta_like(path, label_name):
        seqs = []
        with open(path, 'r') as f:
            cur_id, cur_seq = None, []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if cur_id is not None:
                        seqs.append((cur_id, ''.join(cur_seq), label_name))
                    cur_id, cur_seq = line[1:], []
                else:
                    cur_seq.append(line.replace(' ', '').upper())
            if cur_id is not None:
                seqs.append((cur_id, ''.join(cur_seq), label_name))
        return pd.DataFrame(seqs, columns=['id', 'sequence', 'label'])
    base_dir = Path("Acidophilic-main/Dataset")
    train_files = {base_dir / "Positive.txt": 1, base_dir / "Negative.txt": 0}
    full_df = pd.concat([parse_fasta_like(p, l) for p, l in train_files.items()], ignore_index=True)
    full_df = full_df.reset_index(drop=True)
    print("Full size:", full_df.shape, "Label counts:", full_df["label"].value_counts().to_dict())


    ind_files = {base_dir / "Ind-positive.txt": 1, base_dir / "Ind-negative.txt": 0}  
    ind_df = pd.concat([parse_fasta_like(p, l) for p, l in ind_files.items()], ignore_index=True)
    ind_df = ind_df.reset_index(drop=True)
    print("Independent test size:", ind_df.shape, "Label counts:", ind_df["label"].value_counts().to_dict())


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    metric_keys = ["acc", "sn", "sp", "mcc", "f1", "auc"]
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"]), start=1):
        train_subset = full_df.iloc[train_idx].reset_index(drop=True)
        val_subset   = full_df.iloc[val_idx].reset_index(drop=True)
        fold_train_df = train_subset                
        fold_test_df  = ind_df                       
    
        print("Fold train:", train_subset.shape, "val:", val_subset.shape, "ind_test:", ind_df.shape)

        train_dataset = SequenceDatasetHF(train_subset, 'sequence', 'label', max_length=max_length)
        valid_dataset = SequenceDatasetHF(val_subset, 'sequence', 'label', max_length=max_length)
        test_dataset  = SequenceDatasetHF(ind_df, 'sequence', 'label', max_length=max_length)

        num_labels = 2

        model, tokenizer = initialize_model(
            model_name=model_name,
            num_labels=num_labels,
            use_peft=use_peft,
            peft_config=custom_peft_config,
        )
        model = model.to(device)

        data_collator = SequenceCollator(tokenizer, regression=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        steps_per_epoch = int(np.ceil(len(train_loader) / max(1, gradient_accumulation_steps)))
        total_steps = steps_per_epoch * num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        patience_counter = 0

        best_path = f"{out_dir}/best_model_fold{fold}.pt"

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            print(f"Train Loss: {train_loss:.4f}")

            val_loss, val_logits, val_labels = evaluate_model(model, valid_loader, device)
            val_metrics = compute_metrics_classification(val_logits, val_labels)
            print(f"Val Loss: {val_loss:.4f}")
            print_metrics(val_metrics, "Val ")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_finetuned_weights(model, best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        model, tokenizer = initialize_model(
            model_name=model_name,
            num_labels=num_labels,
            use_peft=use_peft,
            peft_config=custom_peft_config,
        )
        model = model.to(device)
        load_finetuned_weights(model, best_path, device)

        _, best_val_logits, best_val_labels = evaluate_model(model, valid_loader, device)
        T = fit_temperature_scaling(best_val_logits, best_val_labels)
        T *= float(temp_scale_multiplier)

        test_loss, test_logits, test_labels = evaluate_model(model, test_loader, device)
        test_logits_cal = apply_temperature(test_logits, T)
        test_metrics = compute_metrics_classification(test_logits_cal, test_labels)

        print(f"\nFold {fold} Test Loss (raw model loss): {test_loss:.4f}")
        print_metrics(test_metrics, f"Fold-Test (TempScaled, T={T:.2f}) ")

        visualize_umap(
            model=model,
            data_loader=test_loader,
            device=device,
            save_path=f"{out_dir}/umap.pdf"
        )

        plot_confusion_matrix(
            logits=test_logits_cal,
            labels=test_labels,
            out_pdf=f"{out_dir}/confusion_matrix.pdf"
        )

        df_probs = save_test_probabilities(
            logits=test_logits_cal,
            labels=test_labels,
            out_csv=f"{out_dir}/probs.csv",
            ids=fold_test_df["id"].values
        )
        plot_prediction_confidence_density(
            df_probs=df_probs,
            out_pdf=f"{out_dir}/confidence_density.pdf",
            method_name=peft_method.upper()
        )

        row = {"fold": fold, "T": float(T), "raw_test_loss": float(test_loss)}
        row.update(test_metrics)
        fold_metrics.append(row)

        pd.DataFrame(fold_metrics).to_csv(f"{out_dir}/metrics.csv", index=False)

    dfm = pd.DataFrame(fold_metrics)

    mean_row = {k: dfm[k].mean() for k in metric_keys}
    var_row = {k: dfm[k].var(ddof=1) for k in metric_keys}

    summary_df = pd.DataFrame([
        {"stat": "mean", **mean_row},
        {"stat": "var", **var_row},
    ])
    summary_df.to_csv(f"{out_dir}/metrics_mean_var.csv", index=False)

    print("\n" + "=" * 70)
    print("10-FOLD SUMMARY (TempScaled)")
    print("=" * 70)
    print("MEAN:")
    for k in metric_keys:
        print(f"  {k}: {mean_row[k]:.6f}")
    print("VAR:")
    for k in metric_keys:
        print(f"  {k}: {var_row[k]:.6f}")

    return dfm, summary_df


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="10-fold CV training for protein classification")
    parser.add_argument("--model_path", type=str, default="ESMC",
                        help="Path/name of the model to train")
    parser.add_argument("--use_peft", action="store_true", default=True,
                        help="Whether to use PEFT for fine-tuning")
    parser.add_argument("--peft_method", type=str, default="c3a",
                    choices=["c3a", "lora", "miss", "oft", "randlora", "shira"],
                    help="Which PEFT method to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4,  # Others: 5e-5
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum length of input sequences")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--splits", type=int, default=10,
                        help="Number of CV folds")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("CV TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Using PEFT: {args.use_peft}")
    print(f"PEFT method: {args.peft_method}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")
    print(f"Early stopping patience: {args.patience}")
    print(f"CV splits: {args.splits}")
    print("=" * 50 + "\n")

    df_folds, df_summary = train_classification_model_10fold(
        model_name=args.model_path,
        use_peft=args.use_peft,
        peft_method=args.peft_method,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum,
        patience=args.patience,
        n_splits=args.splits,
    )

    print("\nDone! Outputs saved in outputs")

