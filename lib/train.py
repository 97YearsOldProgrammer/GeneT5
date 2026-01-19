import torch
import torch.nn as nn
from pathlib import Path


################################
#####  Training Functions  #####
################################


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch with BF16 mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    num_steps  = 0
    
    # BF16 for training (model stays FP32, compute in BF16)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(**batch) if not hasattr(model, 'forward_finetune') else model(**batch)
            
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs.loss
            
            loss = loss / grad_accum
        
        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1
        
        total_loss += loss.item() * grad_accum
    
    return total_loss / len(dataloader)


def train_epoch_seq2seq(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch - seq2seq with BF16 mixed precision."""
    model.train()
    total_loss = 0
    
    # BF16 for training (model stays FP32, compute in BF16)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(
                encoder_input_ids = batch["input_ids"],
                decoder_input_ids = batch["labels"][:, :-1],
                labels            = batch["labels"][:, 1:],
            )
            
            loss = outputs["loss"] / grad_accum
        
        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set with BF16."""
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**batch)
                
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs.loss
            
            total_loss += loss.item()
            
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                preds  = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += batch["labels"].size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else None
    
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate_seq2seq(model, dataloader, device):
    """Evaluate seq2seq model with BF16."""
    model.eval()
    total_loss = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(
                    encoder_input_ids = batch["input_ids"],
                    decoder_input_ids = batch["labels"][:, :-1],
                    labels            = batch["labels"][:, 1:],
                )
            
            total_loss += outputs["loss"].item()
    
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss


##################################
#####  Checkpoint Functions  #####
##################################


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cpu"):
    """Load model, optimizer, and scheduler states from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch         = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("config", {}).get("best_val_loss", float('inf'))
    
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return {
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "config":        checkpoint.get("config", {})
    }


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None):
    """Save model, optimizer, and scheduler states."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    
    if config:
        checkpoint["config"] = config
    
    torch.save(checkpoint, save_path)


##################################
#####  Model Setup Functions #####
##################################


def setup_gene_prediction_model(model_path, tokenizer, device):
    """Setup model for gene prediction task."""
    from transformers import EncoderDecoderModel
    
    base_model = EncoderDecoderModel.from_pretrained(model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask, labels=None):
            return self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
    
    model = ModelWrapper(base_model)
    model.to(device)
    return model


def setup_rna_classification_model(model_path, tokenizer, num_classes, device, dropout=0.1):
    """Setup model for RNA classification task."""
    from transformers import AutoModel
    
    encoder     = AutoModel.from_pretrained(model_path)
    hidden_size = encoder.config.hidden_size
    encoder.resize_token_embeddings(len(tokenizer))
    
    class ClassificationWrapper(nn.Module):
        def __init__(self, encoder, hidden_size, num_classes, dropout):
            super().__init__()
            self.encoder    = encoder
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)
        
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden  = outputs.last_hidden_state
            pooled  = hidden[:, 0, :]
            pooled  = self.dropout(pooled)
            logits  = self.classifier(pooled)
            
            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss    = loss_fn(logits, labels)
            
            return {"loss": loss, "logits": logits}
    
    model = ClassificationWrapper(encoder, hidden_size, num_classes, dropout)
    model.to(device)
    return model


def prepare_tokenizer(model_path, special_tokens=None):
    """Load and prepare tokenizer with special tokens."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if special_tokens is None:
        special_tokens = ["[GENE]", "[CLS]"]
    
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"Added tokens: {new_tokens}")
    
    return tokenizer


def prepare_optimizer_scheduler(model, train_loader, lr, weight_decay, 
                                epochs, grad_accum, warmup_ratio, scheduler_type="linear"):
    """Prepare optimizer and scheduler."""
    from torch.optim  import AdamW
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    
    total_steps  = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    return optimizer, scheduler


##################################
#####  Utility Functions     #####
##################################


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")