import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, layer2=False, layer3=False, layer4=False):
        super(ResNetFeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        def hook(module, input, output):
            self.features.append(output)
            
        self.layer2_enabled = layer2
        self.layer3_enabled = layer3
        self.layer4_enabled = layer4
        
        if self.layer2_enabled:
            self.model.layer2[-1].register_forward_hook(hook)
        if self.layer3_enabled:
            self.model.layer3[-1].register_forward_hook(hook)
        if self.layer4_enabled:
            self.model.layer4[-1].register_forward_hook(hook)

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]
        self.resize = nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)

        return patch

class FeatCAE(nn.Module):
    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]
        
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AnomalyDetector:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.config['model']['backbone']['input_size'])),
            transforms.ToTensor()
        ])
        
        self.backbone = None
        self.model = None
        self.best_threshold = None
        self.heat_map_max = None
        self.heat_map_min = None
        self.verbose = self.config['training']['verbose']
        
    def setup_models(self):
        self.backbone = ResNetFeatureExtractor(
            layer2=self.config['model']['backbone']['layers']['layer2'],
            layer3=self.config['model']['backbone']['layers']['layer3'],
            layer4=self.config['model']['backbone']['layers']['layer4']
        ).to(self.device)
        
        self.model = FeatCAE(
            in_channels=self.config['model']['autoencoder']['in_channels'],
            latent_dim=self.config['model']['autoencoder']['latent_dim'],
            is_bn=self.config['model']['autoencoder']['batch_norm']
        ).to(self.device)
        
    def load_data(self):
        train_data = ImageFolder(root=self.config['paths']['train_path'], transform=self.transform)
        train_size = int(self.config['training']['train_test_split'] * len(train_data))
        test_size = len(train_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        train_loss = []
        validation_loss = []
        
        for epoch in tqdm(range(self.config['training']['num_epochs'])):
            self.model.train()
            for data, _ in self.train_loader:
                with torch.no_grad():
                    features = self.backbone(data.to(self.device))
                output = self.model(features)
                loss = criterion(output, features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss.append(loss.item())
            
            self.model.eval()
            val_loss_sum = 0.0
            num_batches = 0
            with torch.no_grad():
                for data, _ in self.val_loader:
                    features = self.backbone(data.to(self.device))
                    output = self.model(features)
                    val_loss = criterion(output, features)
                    val_loss_sum += val_loss.item()
                    num_batches += 1
            validation_loss.append(val_loss_sum / num_batches)
            
            if epoch % self.verbose == 0:
                print(f'Epoch [{epoch + 1}/{self.config["training"]["num_epochs"]}], '
                      f'Loss: {loss.item():.4f}, Validation Loss: {validation_loss[-1]:.4f}')
                
        self._calculate_threshold()
        return train_loss, validation_loss
    
    def _decision_function(self, segm_map):
        mean_top_10_values = []
        for map in segm_map:
            flattened_tensor = map.reshape(-1)
            sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)
            mean_top_10_value = sorted_tensor[:10].mean()
            mean_top_10_values.append(mean_top_10_value)
        return torch.stack(mean_top_10_values)
    
    def _calculate_threshold(self):
        recon_errors = []
        self.model.eval()
        for data, _ in self.train_loader:
            with torch.no_grad():
                features = self.backbone(data.to(self.device))
                recon = self.model(features)
                segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
                anomaly_score = self._decision_function(segm_map)
                recon_errors.append(anomaly_score)
        
        recon_errors = torch.cat(recon_errors).cpu().numpy()
        self.best_threshold = np.mean(recon_errors) + self.config['training']['threshold_std_multiplier'] * np.std(recon_errors)
        self.heat_map_max = np.max(recon_errors)
        self.heat_map_min = np.min(recon_errors)
    
    def evaluate(self, test_path):
        test_path = Path(test_path)
        y_true, y_pred, y_score = [], [], []
        
        self.model.eval()
        self.backbone.eval()
        
        for path in test_path.glob('*/*.jpg'):
            fault_type = path.parts[-2]
            test_image = self.transform(Image.open(path)).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                features = self.backbone(test_image)
                recon = self.model(features)
            
            segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
            y_score_image = self._decision_function(segm_map)
            y_pred_image = 1 * (y_score_image >= self.best_threshold)
            y_true_image = 0 if fault_type == 'good' else 1
            
            y_true.append(y_true_image)
            y_pred.append(y_pred_image.cpu().numpy())
            y_score.append(y_score_image.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred), np.array(y_score)
    
    def visualize_results(self, test_path):
        test_path = Path(test_path)
        output_path = Path(self.config['paths']['output_path'])
        output_path.mkdir(exist_ok=True)
        
        for idx, path in enumerate(test_path.glob('*/*.jpg')):
            fault_type = path.parts[-2]
            test_image = self.transform(Image.open(path)).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                features = self.backbone(test_image)
                recon = self.model(features)
            
            segm_map = ((features - recon) ** 2).mean(axis=(1))
            y_score_image = self._decision_function(segm_map)
            y_pred_image = 1 * (y_score_image >= self.best_threshold)
            class_label = ['OK', 'NOK']
            
            if fault_type in ['bad', 'good']:
                plt.figure(figsize=tuple(self.config['visualization']['heatmap']['figure_size']))
                
                plt.subplot(1, 3, 1)
                plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
                plt.title(f'Image type: {fault_type}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                heat_map = segm_map.squeeze().cpu().numpy()
                heat_map = cv2.resize(heat_map, tuple(self.config['model']['backbone']['input_size']))
                plt.imshow(heat_map, cmap=self.config['visualization']['heatmap']['colormap'],
                          vmin=self.heat_map_min, vmax=self.heat_map_max * 3)
                plt.title(f'Anomaly score: {y_score_image[0].cpu().numpy() / self.best_threshold:0.4f} || {class_label[y_pred_image[0]]}')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
                sns.heatmap(heat_map, alpha=self.config['visualization']['heatmap']['alpha'],
                           vmin=self.heat_map_min, vmax=self.heat_map_max,
                           cmap=self.config['visualization']['heatmap']['colormap'])
                plt.axis('off')
                
                plt.savefig(output_path / f'result_{idx}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_model(self):
        save_path = Path(self.config['paths']['model_save_path'])
        save_path.mkdir(exist_ok=True)
        
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_threshold': self.best_threshold,
            'heat_map_max': self.heat_map_max,
            'heat_map_min': self.heat_map_min
        }, save_path / 'model.pth')
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_threshold = checkpoint['best_threshold']
        self.heat_map_max = checkpoint['heat_map_max']
        self.heat_map_min = checkpoint['heat_map_min']
