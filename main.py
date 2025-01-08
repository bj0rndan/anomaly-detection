import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from detector import AnomalyDetector

def plot_training_curves(train_loss, val_loss, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path / 'training_curves.png')
    plt.close()

def plot_roc_curve(y_true, y_score, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path / 'roc_curve.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'full'],
                        help='Operation mode: train, evaluate, or full (both)')
    parser.add_argument('--test_path', type=str,
                        help='Path to test data (required for evaluate and full modes)')
    parser.add_argument('--model_path', type=str,
                        help='Path to save/load model. Load is used when --mode is evaluate')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = AnomalyDetector(args.config)
    detector.setup_models()
    
    if args.mode in ['train', 'full']:
        print("Loading data...")
        detector.load_data()
        
        print("Training model...")
        train_loss, val_loss = detector.train()
        
        print("Saving training curves...")
        plot_training_curves(train_loss, val_loss, output_path)
        
        if args.model_path:
            print(f"Saving model in {args.model_path}")
            detector.save_model()
        print("Processed is finished, everyone is happy.")
        print(f"Find trained model at: {args.model_path} /n Find output images at {args.output_path}")

    if args.mode in ['evaluate', 'full']:
        if not args.test_path:
            raise ValueError("Test path is required for evaluation!")            
        if args.model_path and args.mode == 'evaluate':
            print("Loading model from {args.model_path}")
            detector.load_model(args.model_path)
        
        print("Evaluating model...")
        y_true, y_pred, y_score = detector.evaluate(args.test_path)
        
        print("Generating evaluation plots...")
        plot_roc_curve(y_true, y_score, output_path)
        
        print("Visualizing results...")
        detector.visualize_results(args.test_path)

        print("Processed is finished, everyone is happy.")
        print(f"Find trained model at: {args.model_path} /n Find output images at {args.output_path}")

if __name__ == "__main__":
    main()