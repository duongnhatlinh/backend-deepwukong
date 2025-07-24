#!/usr/bin/env python3
"""
Model setup script
Helps setup the DeepWukong model and configurations
"""

import sys
import os
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

def setup_deepwukong_structure():
    """Setup DeepWukong directory structure"""
    print("📁 Setting up DeepWukong directory structure...")
    
    # Create deepwukong directories
    deepwukong_dir = Path("deepwukong")
    deepwukong_dir.mkdir(exist_ok=True)
    
    src_dir = deepwukong_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    configs_dir = deepwukong_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    joern_dir = deepwukong_dir / "joern"
    joern_dir.mkdir(exist_ok=True)
    
    print("✅ DeepWukong directory structure created")
    
    return deepwukong_dir

def create_sample_config():
    """Create a sample DeepWukong config"""
    config_content = """seed: 42
    num_workers: 2
    log_offline: false

    # preprocess keys
    joern_path: "joern/joern-parse"
    split_token: false

    # data keys
    data_folder: "data"
    save_every_epoch: 1
    val_every_epoch: 1
    log_every_epoch: 10
    progress_bar_refresh_rate: 1

    dataset:
    name: UAV
    token:
        max_parts: 16
        is_wrapped: false
        is_splitted: false
        vocabulary_size: 190000

    gnn:
    name: "gcn"
    w2v_path: "${data_folder}/${dataset.name}/w2v.wv"
    embed_size: 256
    hidden_size: 256
    pooling_ratio: 0.8
    drop_out: 0.5
    n_hidden_layers: 3
    n_head: 3
    n_gru: 3
    edge_sample_ratio: 0.8
    rnn:
        hidden_size: 256
        num_layers: 1
        drop_out: 0.5
        use_bi: true
        activation: relu

    classifier:
    hidden_size: 512
    n_hidden_layers: 2
    n_classes: 2
    drop_out: 0.5

    hyper_parameters:
    vector_length: 128
    n_epochs: 50
    patience: 10
    batch_size: 32
    test_batch_size: 32
    reload_dataloader: true
    clip_norm: 5
    val_every_step: 1.0
    log_every_n_steps: 50
    progress_bar_refresh_rate: 1
    resume_from_checkpoint: null
    shuffle_data: true
    optimizer: "Adam"
    nesterov: true
    learning_rate: 0.005
    weight_decay: 0
    decay_gamma: 0.95
    gradient_accumulation_steps: 2
    """
    
    config_path = Path("deepwukong/configs/dwk.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Sample config created at {config_path}")

def copy_deepwukong_source():
    """Instructions for copying DeepWukong source"""
    print("\n📋 MANUAL STEPS REQUIRED:")
    print("=" * 50)
    print("1. Copy your DeepWukong source code to 'deepwukong/src/'")
    print("   Required files:")
    print("   - src/enhanced_detect.py")
    print("   - src/models/")
    print("   - src/datas/")
    print("   - src/vocabulary.py")
    print("   - src/utils.py")
    print("   - src/preprocess/")
    print()
    print("2. Copy your trained model checkpoint to 'storage/models/'")
    print(f"   Expected path: {settings.DEEPWUKONG_MODEL_PATH}")
    print()
    print("3. Copy Joern binaries to 'deepwukong/joern/'")
    print(f"   Expected path: {settings.JOERN_PATH}")
    print()
    print("4. Update config paths in .env file if needed")

def check_requirements():
    """Check if required components exist"""
    print("\n🔍 Checking requirements...")
    
    checks = [
        ("DeepWukong source", Path("deepwukong/src/enhanced_detect.py")),
        ("Model checkpoint", Path(settings.DEEPWUKONG_MODEL_PATH)),
        ("Config file", Path(settings.DEEPWUKONG_CONFIG_PATH)),
        ("Joern binary", Path(settings.JOERN_PATH)),
    ]
    
    all_good = True
    for name, path in checks:
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (missing)")
            all_good = False
    
    if all_good:
        print("\n🎉 All requirements satisfied!")
    else:
        print(f"\n⚠️  Some requirements missing. See instructions above.")
    
    return all_good

def main():
    """Main setup function"""
    print("🚀 DeepWukong Model Setup")
    print("=" * 30)
    
    # Setup directory structure
    setup_deepwukong_structure()
    
    # Create sample config
    create_sample_config()
    
    # Show manual steps
    copy_deepwukong_source()
    
    # Check requirements
    check_requirements()
    
    print("\n🏁 Setup complete!")
    print("Run 'python run.py' to start the server after completing manual steps.")

if __name__ == "__main__":
    main()