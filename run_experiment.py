#!/usr/bin/env python3
"""
run_experiment.py

High-level experiment runner for seismic frequency recovery experiments.

This script provides a unified interface for running frequency recovery experiments
with different configurations, datasets, and model variants. It handles experiment
orchestration, configuration management, and results aggregation.

Usage:
    python run_experiment.py --experiment-name basic_recovery --data-dir /path/to/data
    python run_experiment.py --config experiments/spectral_config.yaml
    python run_experiment.py --sweep --param-grid param_grid.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.core.logging import get_logger
from src.core.utils import PML_kw

__all__ = ["ExperimentRunner", "main"]


class ExperimentRunner:
    """
    Orchestrates seismic frequency recovery experiments.
    
    Features:
    - Configuration management and validation
    - Multi-run experiments with different parameters
    - Results tracking and aggregation
    - Automatic experiment logging and checkpointing
    - Integration with existing BaseTrainer infrastructure
    """
    
    def __init__(self, experiment_name: str, base_config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.base_config = base_config
        self.logger = get_logger(f"ExperimentRunner.{experiment_name}")
        
        # Setup experiment directory
        self.experiment_dir = Path("experiments") / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.results = []
        self.config_log = []
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate experiment configuration."""
        required_keys = ["data_dir", "epochs", "batch_size"]
        
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing required config key: {key}")
                return False
        
        # Validate data directory exists
        data_dir = Path(config["data_dir"])
        if not data_dir.exists():
            self.logger.error(f"Data directory does not exist: {data_dir}")
            return False
        
        # Check for required subdirectories
        required_subdirs = ["HP", "LP", "full"]
        for subdir in required_subdirs:
            if not (data_dir / subdir).exists():
                self.logger.warning(f"Missing data subdirectory: {subdir}")
        
        return True
    
    def prepare_run_config(self, config: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Prepare configuration for a specific run."""
        run_config = config.copy()
        
        # Add run-specific settings
        run_config["run_id"] = run_id
        run_config["save_dir"] = str(self.experiment_dir / f"run_{run_id:03d}")
        
        # Ensure save directory exists
        Path(run_config["save_dir"]).mkdir(parents=True, exist_ok=True)
        
        return run_config
    
    def build_training_command(self, config: Dict[str, Any]) -> List[str]:
        """Build command line for training script."""
        cmd = [sys.executable, "train_spectralunet.py"]
        
        # Add required arguments
        cmd.extend(["--data-dir", str(config["data_dir"])])
        cmd.extend(["--epochs", str(config["epochs"])])
        cmd.extend(["--batch-size", str(config["batch_size"])])
        cmd.extend(["--save-dir", str(config["save_dir"])])
        
        # Add optional arguments
        optional_args = {
            "learning_rate": "--learning-rate",
            "weight_decay": "--weight-decay",
            "encoder": "--encoder",
            "encoder_weights": "--encoder-weights",
            "penalty_weight": "--penalty-weight",
            "magnitude_weight": "--magnitude-weight",
            "phase_weight": "--phase-weight",
            "device": "--device",
            "num_workers": "--num-workers",
            "val_frequency": "--val-frequency"
        }
        
        for config_key, cmd_arg in optional_args.items():
            if config_key in config:
                cmd.extend([cmd_arg, str(config[config_key])])
        
        return cmd
    
    def run_single_experiment(self, config: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Run a single experiment with given configuration."""
        self.logger.info(f"Starting run {run_id} with config: {config}")
        
        # Prepare run-specific configuration
        run_config = self.prepare_run_config(config, run_id)
        
        # Save configuration for this run
        config_path = Path(run_config["save_dir"]) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)
        
        # Build and execute training command
        cmd = self.build_training_command(run_config)
        self.logger.info(f"Executing command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=run_config.get("timeout", 3600)  # 1 hour default timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Process results
            if result.returncode == 0:
                self.logger.info(f"Run {run_id} completed successfully in {duration:.2f}s")
                status = "completed"
                error_message = None
            else:
                self.logger.error(f"Run {run_id} failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                status = "failed"
                error_message = result.stderr
            
            # Extract metrics from training output (basic parsing)
            metrics = self.parse_training_output(result.stdout, result.stderr)
            
            run_result = {
                "run_id": run_id,
                "config": run_config,
                "status": status,
                "duration": duration,
                "metrics": metrics,
                "error_message": error_message,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            # Save run results
            results_path = Path(run_config["save_dir"]) / "results.json"
            with open(results_path, 'w') as f:
                json.dump(run_result, f, indent=2, default=str)
            
            return run_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Run {run_id} timed out after {duration:.2f}s")
            
            return {
                "run_id": run_id,
                "config": run_config,
                "status": "timeout",
                "duration": duration,
                "metrics": {},
                "error_message": "Process timed out"
            }
        
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"Run {run_id} failed with exception: {str(e)}")
            
            return {
                "run_id": run_id,
                "config": run_config,
                "status": "error",
                "duration": duration,
                "metrics": {},
                "error_message": str(e)
            }
    
    def parse_training_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse training output to extract metrics."""
        metrics = {}
        
        try:
            # Look for final validation loss in stdout
            lines = stdout.split('\n')
            for line in lines:
                if "Valid Total Loss:" in line:
                    # Extract final validation loss
                    parts = line.split("Valid Total Loss:")
                    if len(parts) > 1:
                        try:
                            final_val_loss = float(parts[1].strip())
                            metrics["final_validation_loss"] = final_val_loss
                        except ValueError:
                            pass
                
                if "Saved best model with validation loss:" in line:
                    # Extract best validation loss
                    parts = line.split("validation loss:")
                    if len(parts) > 1:
                        try:
                            best_val_loss = float(parts[1].strip())
                            metrics["best_validation_loss"] = best_val_loss
                        except ValueError:
                            pass
        
        except Exception as e:
            self.logger.warning(f"Failed to parse training output: {str(e)}")
        
        return metrics
    
    def run_parameter_sweep(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Run parameter sweep experiment."""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Running parameter sweep with {len(combinations)} combinations")
        
        all_results = []
        
        for i, combination in enumerate(combinations):
            # Create configuration for this combination
            config = self.base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            # Run experiment
            result = self.run_single_experiment(config, i + 1)
            all_results.append(result)
            
            # Log progress
            completed = i + 1
            self.logger.info(f"Completed {completed}/{len(combinations)} parameter combinations")
        
        return all_results
    
    def run_experiment(self, param_grid: Optional[Dict[str, List[Any]]] = None) -> List[Dict[str, Any]]:
        """Run complete experiment."""
        if param_grid:
            # Parameter sweep
            results = self.run_parameter_sweep(param_grid)
        else:
            # Single run
            results = [self.run_single_experiment(self.base_config, 1)]
        
        # Save aggregate results
        self.save_experiment_summary(results)
        
        return results
    
    def save_experiment_summary(self, results: List[Dict[str, Any]]):
        """Save experiment summary and results."""
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "base_config": self.base_config,
            "total_runs": len(results),
            "successful_runs": len([r for r in results if r["status"] == "completed"]),
            "failed_runs": len([r for r in results if r["status"] != "completed"]),
            "results": results
        }
        
        # Save summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save best result if available
        completed_results = [r for r in results if r["status"] == "completed" and "best_validation_loss" in r["metrics"]]
        if completed_results:
            best_result = min(completed_results, key=lambda r: r["metrics"]["best_validation_loss"])
            best_path = self.experiment_dir / "best_result.json"
            with open(best_path, 'w') as f:
                json.dump(best_result, f, indent=2, default=str)
            
            self.logger.info(f"Best validation loss: {best_result['metrics']['best_validation_loss']:.6f}")
        
        self.logger.info(f"Experiment summary saved to: {summary_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config(**kwargs) -> Dict[str, Any]:
    """Create default experiment configuration."""
    return {
        "data_dir": PML_kw("data_dir", kwargs, "/path/to/data"),
        "epochs": PML_kw("epochs", kwargs, 50),
        "batch_size": PML_kw("batch_size", kwargs, 8),
        "learning_rate": PML_kw("learning_rate", kwargs, 1e-3),
        "weight_decay": PML_kw("weight_decay", kwargs, 1e-4),
        "encoder": PML_kw("encoder", kwargs, "resnet34"),
        "encoder_weights": PML_kw("encoder_weights", kwargs, "imagenet"),
        "penalty_weight": PML_kw("penalty_weight", kwargs, 1.0),
        "magnitude_weight": PML_kw("magnitude_weight", kwargs, 0.5),
        "phase_weight": PML_kw("phase_weight", kwargs, 0.3),
        "device": PML_kw("device", kwargs, "auto"),
        "num_workers": PML_kw("num_workers", kwargs, 4),
        "val_frequency": PML_kw("val_frequency", kwargs, 5)
    }


def main():
    """Main experiment runner function."""
    parser = argparse.ArgumentParser(description="Run seismic frequency recovery experiments")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, default="spectral_recovery",
                      help="Name of the experiment")
    parser.add_argument("--config", type=str,
                      help="Path to YAML configuration file")
    
    # Parameter sweep
    parser.add_argument("--sweep", action="store_true",
                      help="Run parameter sweep")
    parser.add_argument("--param-grid", type=str,
                      help="Path to parameter grid YAML file")
    
    # Quick configuration overrides
    parser.add_argument("--data-dir", type=str,
                      help="Path to data directory")
    parser.add_argument("--epochs", type=int,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,
                      help="Training batch size")
    parser.add_argument("--learning-rate", type=float,
                      help="Learning rate")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger("run_experiment")
    logger.info(f"Starting experiment runner with args: {args}")
    
    try:
        # Load or create configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = create_default_config()
            logger.info("Using default configuration")
        
        # Apply command line overrides
        if args.data_dir:
            config["data_dir"] = args.data_dir
        if args.epochs:
            config["epochs"] = args.epochs
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        
        # Create experiment runner
        runner = ExperimentRunner(args.experiment_name, config)
        
        # Validate configuration
        if not runner.validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Load parameter grid if specified
        param_grid = None
        if args.sweep:
            if args.param_grid:
                param_grid = load_config(args.param_grid)
                logger.info(f"Loaded parameter grid from: {args.param_grid}")
            else:
                # Default parameter grid
                param_grid = {
                    "learning_rate": [1e-3, 5e-4, 1e-4],
                    "penalty_weight": [0.5, 1.0, 2.0],
                    "magnitude_weight": [0.3, 0.5, 0.7]
                }
                logger.info("Using default parameter grid")
        
        # Run experiment
        logger.info("Starting experiment execution...")
        results = runner.run_experiment(param_grid)
        
        # Summary
        successful_runs = len([r for r in results if r["status"] == "completed"])
        logger.info(f"Experiment completed: {successful_runs}/{len(results)} runs successful")
        
        if successful_runs > 0:
            logger.info(f"Results saved in: {runner.experiment_dir}")
        else:
            logger.error("No successful runs completed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
