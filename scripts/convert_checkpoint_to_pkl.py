"""Convert a PyTorch Lightning checkpoint to a Darts .pkl model file.

This script loads a trained N-BEATS model from the checkpoint directory
and saves it as a .pkl file that can be used with predict and evaluate commands.

Usage:
    python scripts/convert_checkpoint_to_pkl.py
    python scripts/convert_checkpoint_to_pkl.py --checkpoint best  # Use best checkpoint
    python scripts/convert_checkpoint_to_pkl.py --output models/my_nbeats.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from darts.models import NBEATSModel as DartsNBEATS
from loguru import logger


def find_checkpoint_file(checkpoint_dir: Path, checkpoint_type: str = "best") -> Path:
    """Find the checkpoint file in the directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_type: Type of checkpoint to use ("best" or "last")

    Returns:
        Path to the checkpoint file

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    checkpoints_subdir = checkpoint_dir / "checkpoints"

    if not checkpoints_subdir.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_subdir}"
        )

    # List all checkpoint files
    ckpt_files = list(checkpoints_subdir.glob("*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_subdir}")

    # Filter by checkpoint type
    if checkpoint_type == "best":
        best_ckpts = [f for f in ckpt_files if "best" in f.name.lower()]
        if best_ckpts:
            # Sort by validation loss (lowest first)
            checkpoint_file = sorted(best_ckpts)[0]
        else:
            logger.warning("No 'best' checkpoint found, using 'last' instead")
            checkpoint_file = next(
                (f for f in ckpt_files if "last" in f.name.lower()), ckpt_files[0]
            )
    else:  # last
        last_ckpts = [f for f in ckpt_files if "last" in f.name.lower()]
        checkpoint_file = last_ckpts[0] if last_ckpts else ckpt_files[-1]

    return checkpoint_file


def convert_checkpoint_to_pkl(
    checkpoint_dir: Path,
    output_path: Path,
    checkpoint_type: str = "best",
) -> None:
    """Convert a checkpoint to a .pkl file.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        output_path: Path where to save the .pkl file
        checkpoint_type: Type of checkpoint to use ("best" or "last")
    """
    logger.info(f"Converting checkpoint from {checkpoint_dir}")

    # Find the checkpoint file
    checkpoint_file = find_checkpoint_file(checkpoint_dir, checkpoint_type)
    logger.info(f"Using checkpoint: {checkpoint_file.name}")

    # Darts saves a _model.pth.tar file alongside checkpoints
    # We need to load the model using the checkpoint directory
    try:
        # Load the model from the checkpoint directory
        # Darts NBEATSModel.load() can handle checkpoint directories
        logger.info("Loading model from checkpoint...")
        model = DartsNBEATS.load_from_checkpoint(
            model_name="nbeats",
            work_dir=str(checkpoint_dir.parent),
            best=checkpoint_type == "best",
        )

        logger.info("Model loaded successfully!")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as .pkl file
        logger.info(f"Saving model to {output_path}")
        model.save(str(output_path))

        logger.info(f"✓ Successfully converted checkpoint to {output_path}")
        logger.info("You can now use this model with:")
        logger.info(f"  python -m price_stradamus.cli predict --model {output_path}")
        logger.info(
            f"  python -m price_stradamus.cli evaluate --model-path {output_path}"
        )

    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {e}")
        logger.error("Trying alternative loading method...")

        # Alternative method: Load using the _model.pth.tar file
        try:
            pth_file = checkpoint_dir / "_model.pth.tar"
            if pth_file.exists():
                logger.info(f"Loading from {pth_file}")
                model = DartsNBEATS.load(str(pth_file))

                # Create output directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as .pkl file
                logger.info(f"Saving model to {output_path}")
                model.save(str(output_path))

                logger.info(f"✓ Successfully converted checkpoint to {output_path}")
                logger.info("You can now use this model with:")
                logger.info(
                    f"  python -m price_stradamus.cli predict --model {output_path}"
                )
                logger.info(
                    f"  python -m price_stradamus.cli evaluate --model-path {output_path}"
                )
            else:
                raise FileNotFoundError(f"Could not find {pth_file}")

        except Exception as e2:
            logger.error(f"Alternative method also failed: {e2}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoint to Darts .pkl file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("modelsResults/checkpoints/nbeats"),
        help="Directory containing the model checkpoints (default: modelsResults/checkpoints/nbeats)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("modelsResults/nbeats_model.pkl"),
        help="Output path for the .pkl file (default: modelsResults/nbeats_model.pkl)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to use: 'best' (lowest val_loss) or 'last' (final epoch)",
    )

    args = parser.parse_args()

    # Validate checkpoint directory exists
    if not args.checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        logger.error(
            "Please provide a valid checkpoint directory with --checkpoint-dir"
        )
        return 1

    # Convert
    try:
        convert_checkpoint_to_pkl(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            checkpoint_type=args.checkpoint,
        )
        return 0
    except Exception:
        logger.exception("Conversion failed")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
