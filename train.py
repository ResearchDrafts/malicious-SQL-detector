"""
Training script for the MLP classifier.
Loads data, trains the model with early stopping, evaluates on test set,
and saves trained weights.

Usage:
    python train.py
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.MLP import SQLSecurityDataset, MLPClassifier, MLPTrainer, plot_confusion_matrix
from pipeline.codebert import UnixCoderEncoder
from pipeline.llm_checker import LLMChecker
from pipeline.encoder import FeatureEncoder
from config import DEVICE, TRAIN_CSV, VAL_CSV, TEST_CSV, FINAL_CLASSIFIER_PATH


def main():
    """
    Main training pipeline.
    """
    device = DEVICE if DEVICE else "cpu"
    print(f"Using device: {device}")

    print("\n=== Initializing Encoders ===")
    try:
        codebert_encoder = UnixCoderEncoder()
        llm_checker = LLMChecker()
        feature_encoder = FeatureEncoder()
        print("Encoders initialized successfully")
    except Exception as exc:
        print(f"Error initializing encoders: {exc}")
        return

    print("\n=== Loading Datasets ===")
    try:
        train_dataset = SQLSecurityDataset(
            TRAIN_CSV, codebert_encoder, llm_checker, feature_encoder
        )
        val_dataset = SQLSecurityDataset(
            VAL_CSV, codebert_encoder, llm_checker, feature_encoder
        )
        test_dataset = SQLSecurityDataset(
            TEST_CSV, codebert_encoder, llm_checker, feature_encoder
        )
        
        print(f"Train set: {len(train_dataset)} samples")
        print(f"Val set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
    except Exception as exc:
        print(f"Error loading datasets: {exc}")
        return

    print("\n=== Creating DataLoaders ===")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n=== Building and Training Model ===")
    model = MLPClassifier()
    trainer = MLPTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        patience=5,
    )

    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
    )

    print("\n=== Saving Model ===")
    model_path = Path(FINAL_CLASSIFIER_PATH)
    trainer.save_checkpoint(model_path)

    print("\n=== Evaluating on Test Set ===")
    metrics = trainer.evaluate(test_loader)

    print("\n=== Test Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    print("\n=== Saving Plots ===")
    trainer.plot_training_history(save_path="./artifacts/training_history.png")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path="./artifacts/confusion_matrix.png"
    )

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
