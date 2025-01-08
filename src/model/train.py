import torch


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
):
    model.train()
    total_loss = 0
    for input_vector, target_eval in train_loader:
        input_vector = input_vector.to(device)
        target_eval = target_eval.to(device)

        with torch.autocast(device_type=str(device), dtype=torch.float16):
            pred_vector = model(input_vector)
            pred_eval = torch.sum(pred_vector, dim=1)
            loss = criterion(pred_eval, target_eval)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

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
