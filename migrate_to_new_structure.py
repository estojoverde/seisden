#!/usr/bin/env python3
"""
migrate_to_new_structure.py

Migration script to transition from the old src/ structure to the new organized structure.

This script:
1. Backs up the old src/ directory
2. Replaces it with the new organized structure
3. Updates import statements in training scripts
4. Provides migration guidance
"""
from __future__ import annotations

import shutil
import os
from pathlib import Path
import argparse

def backup_old_structure(src_dir: Path, backup_dir: Path):
    """Create backup of old src structure."""
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    print(f"Creating backup: {src_dir} -> {backup_dir}")
    shutil.copytree(src_dir, backup_dir)
    print("✓ Backup created successfully")

def migrate_to_new_structure(old_src: Path, new_src: Path, target_src: Path):
    """Replace old structure with new organized structure."""
    # Remove old structure
    if target_src.exists():
        shutil.rmtree(target_src)
    
    # Copy new structure
    print(f"Migrating: {new_src} -> {target_src}")
    shutil.copytree(new_src, target_src)
    print("✓ New structure installed successfully")

def update_training_scripts(base_dir: Path):
    """Update import statements in training scripts."""
    scripts_to_update = [
        "train_spectralunet.py",
        "run_experiment.py", 
        "example_usage.py"
    ]
    
    # Mapping of old imports to new imports
    import_mapping = {
        "from src.trainer import": "from src.core.trainer import",
        "from src.models import": "from src.models.spectral import",
        "from src.dataset_spectral import": "from src.data.dataset_spectral import",
        "from src.losses import": "from src.losses import",
        "from src.losses_spectral import": "from src.losses.losses_spectral import",
        "from src.logging import": "from src.core.logging import",
        "from src.utils import": "from src.core.utils import",
    }
    
    for script_name in scripts_to_update:
        script_path = base_dir / script_name
        if script_path.exists():
            print(f"Updating imports in {script_name}...")
            
            # Read file
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Update imports
            for old_import, new_import in import_mapping.items():
                content = content.replace(old_import, new_import)
            
            # Write back
            with open(script_path, 'w') as f:
                f.write(content)
            
            print(f"✓ Updated {script_name}")
        else:
            print(f"⚠ Script not found: {script_name}")

def create_compatibility_layer(src_dir: Path):
    """Create a compatibility layer for old import paths."""
    compat_file = src_dir / "compat.py"
    
    compat_content = '''# src/compat.py
"""
Compatibility layer for old import paths.

This module provides backward compatibility for code that uses the old
import structure. It should be used temporarily during migration.

Usage:
    # Old style (deprecated)
    from src.models import PML_create_spectral_unet
    
    # New style (preferred)
    from src.models.spectral import PML_create_spectral_unet
"""
import warnings

# Issue deprecation warning
warnings.warn(
    "Using compatibility imports is deprecated. "
    "Please update to the new modular import structure.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export commonly used items for backward compatibility
try:
    # Core
    from .core.utils import PML_kw
    from .core.logging import get_logger
    from .core.trainer import PML_ModelTrainer
    from .core.metrics import PML_snr_lowband, PML_spectral_l2_bands
    
    # Models
    from .models.spectral import PML_SpectralUNet, PML_create_spectral_unet
    from .models.diffusion import PML_SpectralDDPM, PML_build_spectral_ddpm
    
    # Data
    from .data.dataset import PML_NpyPairedSeismic, PML_apply_lowcut_fft
    from .data.dataset_spectral import PML_SeismicFrequencyRecoveryDataset
    
    # Losses
    from .losses.losses import PML_spectral_l2_per_band, PML_lowband_weighted_loss
    from .losses.losses_spectral import (
        PML_SeismicRecoveryLoss, 
        PML_SpectralMSELoss, 
        PML_ReconstructionPenaltyLoss,
        create_spectral_loss
    )
    
    # Features
    from .features.fourier import PML_radial_lf_mask2d
    from .features.layers import FourierLayer, InverseFourierLayer

except ImportError as e:
    warnings.warn(f"Some compatibility imports failed: {e}")

'''
    
    with open(compat_file, 'w') as f:
        f.write(compat_content)
    
    print(f"✓ Created compatibility layer: {compat_file}")

def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate to new src/ structure")
    parser.add_argument("--backup", action="store_true", default=True,
                      help="Create backup of old structure")
    parser.add_argument("--dry-run", action="store_true",
                      help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path("/home/marcelo.silva/Projetos/BaseTrainer/seisden")
    old_src = base_dir / "src"
    new_src = base_dir / "src_new"
    backup_src = base_dir / "src_backup"
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print(f"Would backup: {old_src} -> {backup_src}")
        print(f"Would migrate: {new_src} -> {old_src}")
        print("Would update training script imports")
        print("Would create compatibility layer")
        return
    
    print("🚀 Starting migration to new organized src/ structure...")
    
    try:
        # Step 1: Backup old structure
        if args.backup and old_src.exists():
            backup_old_structure(old_src, backup_src)
        
        # Step 2: Migrate to new structure
        if new_src.exists():
            migrate_to_new_structure(old_src, new_src, old_src)
        else:
            print("❌ New structure not found at src_new/")
            return
        
        # Step 3: Update training scripts
        update_training_scripts(base_dir)
        
        # Step 4: Create compatibility layer
        create_compatibility_layer(old_src)
        
        # Step 5: Cleanup
        if new_src.exists():
            shutil.rmtree(new_src)
            print("✓ Cleaned up temporary src_new directory")
        
        print("\n" + "="*60)
        print("🎉 Migration completed successfully!")
        print("="*60)
        print("\nNew structure:")
        print("src/")
        print("├── core/           # Core utilities, logging, trainer")
        print("├── models/")
        print("│   ├── diffusion/  # Diffusion models (DDPM, UNet blocks)")
        print("│   └── spectral/   # Spectral U-Net models")
        print("├── data/           # Dataset classes")
        print("├── training/       # Training loops and scripts")
        print("├── losses/         # Loss functions")
        print("├── features/       # Feature processing")
        print("└── visualization/  # Plotting and visualization")
        print("\nWhat changed:")
        print("✓ Separated diffusion and spectral model components")
        print("✓ Organized core infrastructure into core/ module")
        print("✓ Grouped related functionality into logical modules")
        print("✓ Updated import statements in training scripts")
        print("✓ Created compatibility layer for gradual migration")
        print("\nNext steps:")
        print("1. Test that training scripts work with new structure")
        print("2. Update any custom scripts to use new import paths")
        print("3. Remove compatibility layer once migration is complete")
        
        if args.backup:
            print(f"4. Remove backup directory when confident: {backup_src}")
    
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
