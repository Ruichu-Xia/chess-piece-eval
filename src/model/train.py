import os

import torch
from tqdm import tqdm


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                num_epochs,
                device,
                scaler):
    model = model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, num_epochs+1):
        train_loss = train_model_single_epoch(
               model,
               train_loader,
               criterion,
               optimizer,
               device,
               scaler
        )
        val_loss = validate_model_single_epoch(
               model,
               val_loader,
               criterion,
               device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch+1) % 1000 == 0 or epoch == num_epochs:
            print(f"[Epoch {epoch}/{num_epochs}]",
                  f"Train Loss: {train_loss:.4f} | ",
                  f"Val Loss: {val_loss:.4f}")

    return model, history


def train_model_single_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        scaler,
        grad_clip
):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for input_vector, target_eval in progress_bar:
        input_vector = input_vector.to(device)
        target_eval = target_eval.to(device)

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            pred_vector = model(input_vector)
            pred_eval = torch.sum(pred_vector, dim=1)
            loss = criterion(pred_eval, target_eval)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Detected NaN or Inf loss. Skipping batch.")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


def validate_model_single_epoch(
        model,
        val_loader,
        criterion,
        device,
):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_vector, target_eval in val_loader:
            input_vector = input_vector.to(device)
            target_eval = target_eval.to(device)

            pred_vector = model(input_vector)
            pred_eval = torch.sum(pred_vector, dim=1)
            loss = criterion(pred_eval, target_eval)

            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def save_checkpoint(epoch, model, history, checkpoint_dir, test_indices=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
        "test_indices": test_indices,
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
