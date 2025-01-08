import gradio as gr
import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from detector import AnomalyDetector
import tempfile
import os
import shutil

class AnomalyDetectionApp:
    def __init__(self, config_path="config.yaml"):
        self.detector = AnomalyDetector(config_path)
        self.detector.setup_models()
        self.current_model_path = None
        
    def train_model(self, train_path, progress=gr.Progress()):
        # Create temporary directory structure for training
        temp_train_dir = Path(tempfile.mkdtemp())
        good_dir = temp_train_dir / "good"
        good_dir.mkdir(parents=True)
        
        # Copy training images to temporary directory
        for file in train_path:
            shutil.copy(file.name, good_dir)
            
        # Update config with temporary path
        self.detector.config['paths']['train_path'] = str(temp_train_dir)
        
        # Load data and train model
        progress(0, desc="Loading data...")
        self.detector.load_data()
        
        progress(0.2, desc="Training model...")
        train_loss, val_loss = self.detector.train()
        
        # Create loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save plot to temporary file
        loss_plot_path = "temp_loss_plot.png"
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Save model
        model_save_dir = Path("saved_models")
        model_save_dir.mkdir(exist_ok=True)
        self.current_model_path = model_save_dir / "latest_model.pth"
        self.detector.save_model()
        
        # Cleanup
        shutil.rmtree(temp_train_dir)
        
        return loss_plot_path, "Model trained successfully!"
    
    def detect_anomalies(self, image_path, progress=gr.Progress()):
        if self.current_model_path is None:
            return None, "Please train or load a model first!"
            
        progress(0.2, desc="Processing image...")
        
        # Load and process image
        image = Image.open(image_path.name)
        test_image = self.detector.transform(image).to(self.detector.device).unsqueeze(0)
        
        with torch.no_grad():
            features = self.detector.backbone(test_image)
            recon = self.detector.model(features)
        
        segm_map = ((features - recon) ** 2).mean(axis=(1))
        y_score_image = self.detector._decision_function(segm_map)
        y_pred_image = 1 * (y_score_image >= self.detector.best_threshold)
        
        progress(0.6, desc="Generating visualization...")
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        heat_map = segm_map.squeeze().cpu().numpy()
        plt.imshow(heat_map, cmap='magma',
                  vmin=self.detector.heat_map_min,
                  vmax=self.detector.heat_map_max * 3)
        plt.title(f'Anomaly Score: {y_score_image[0].cpu().numpy() / self.detector.best_threshold:.4f}')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.imshow(heat_map, cmap='magma', alpha=0.25,
                  vmin=self.detector.heat_map_min,
                  vmax=self.detector.heat_map_max)
        plt.title('Overlay')
        plt.axis('off')
        
        # Save visualization
        result_path = "temp_result.png"
        plt.savefig(result_path, bbox_inches='tight')
        plt.close()
        
        result_text = f"Prediction: {'Anomaly' if y_pred_image[0] else 'Normal'}\n"
        result_text += f"Anomaly Score: {y_score_image[0].cpu().numpy() / self.detector.best_threshold:.4f}"
        
        return result_path, result_text
    
    def load_saved_model(self, model_path, progress=gr.Progress()):
        progress(0.3, desc="Loading model...")
        self.current_model_path = model_path.name
        self.detector.load_model(model_path.name)
        return "Model loaded successfully!"

def create_app():
    app = AnomalyDetectionApp()
    
    with gr.Blocks(title="Industrial Anomaly Detection") as interface:
        gr.Markdown("# Industrial Anomaly Detection System")
        
        with gr.Tab("Train Model"):
            gr.Markdown("### Train New Model")
            with gr.Row():
                train_files = gr.File(
                    file_count="multiple",
                    label="Upload Training Images (Normal samples only)",
                    file_types=["image"]
                )
            with gr.Row():
                train_button = gr.Button("Train Model")
            with gr.Row():
                loss_plot = gr.Image(label="Training Progress")
                train_output = gr.Textbox(label="Training Status")
                
        with gr.Tab("Load Model"):
            gr.Markdown("### Load Existing Model")
            with gr.Row():
                model_file = gr.File(label="Upload Model File (.pth)")
                load_status = gr.Textbox(label="Load Status")
            load_button = gr.Button("Load Model")
            
        with gr.Tab("Detect Anomalies"):
            gr.Markdown("### Anomaly Detection")
            with gr.Row():
                input_image = gr.File(label="Upload Image for Detection")
                detect_button = gr.Button("Detect Anomalies")
            with gr.Row():
                result_image = gr.Image(label="Detection Results")
                result_text = gr.Textbox(label="Detection Details")
        
        # Event handlers
        train_button.click(
            app.train_model,
            inputs=[train_files],
            outputs=[loss_plot, train_output]
        )
        
        load_button.click(
            app.load_saved_model,
            inputs=[model_file],
            outputs=[load_status]
        )
        
        detect_button.click(
            app.detect_anomalies,
            inputs=[input_image],
            outputs=[result_image, result_text]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
