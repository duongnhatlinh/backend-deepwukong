import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from src.models.memory_efficient_vd import MemoryEfficientDeepWukong
from src.datas.datamodules import XFGDataModule
from src.vocabulary import Vocabulary
from src.utils import filter_warnings


def train_memory_efficient_model(config_path: str):
    """Train memory-efficient model for GTX 1650"""
    filter_warnings()
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Set memory-efficient settings
    torch.backends.cudnn.benchmark = True  # Enable for speed (was False)
    torch.backends.cudnn.deterministic = False  # Disable for speed (was True)
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  # Enable for speed (was False)
    
    # Load vocabulary
    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()
    
    # Create data module
    data_module = XFGDataModule(config, vocab)
    
    # Create memory-efficient model
    model = MemoryEfficientDeepWukong(config, vocab, vocab_size, pad_idx)
    
    # Memory-efficient callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_f1",
        filename="uav-gtx1650-{epoch:02d}-{val_f1:.3f}",
        save_top_k=3,
        mode="max",
        save_weights_only=True  # Save memory
    )
    
    early_stopping = EarlyStopping(
        patience=config.hyper_parameters.patience,
        monitor="val_f1",
        mode="max",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor("epoch")
    
    # Memory-efficient trainer
    trainer = L.Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        # gradient_clip_val=config.hyper_parameters.clip_norm,
        # accumulate_grad_batches=config.hyper_parameters.gradient_accumulation_steps,
        precision="16-mixed" if config.hyper_parameters.get('use_mixed_precision') else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=False,  # Disable for speed (was True)
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=TensorBoardLogger("logs", name="uav_gtx1650"),
        log_every_n_steps=config.hyper_parameters.log_every_n_steps,
        check_val_every_n_epoch=1,
        enable_model_summary=False,  # Disable for speed (was True)
        sync_batchnorm=False,  # Not needed for single GPU
        # find_unused_parameters=False,  # Memory optimization
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=0,  # Disable for speed
    )
    
    # Train model
    print("🚀 Starting memory-efficient training for GTX 1650...")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"💾 Effective batch size: {config.hyper_parameters.batch_size * config.hyper_parameters.gradient_accumulation_steps}")
    
    trainer.fit(model, data_module)
    
    # Test with best model
    trainer.test(model, data_module, ckpt_path="best")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    train_memory_efficient_model("configs/dwk_uav_gtx1650.yaml")