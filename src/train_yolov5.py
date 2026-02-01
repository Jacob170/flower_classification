import torch
import torch.nn as nn
from ultralytics import YOLO
import sys

# Add yolov5 to path (if you cloned it)
sys.path.append("../yolov5")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 102
EPOCHS = 20
LEARNING_RATE = 0.001


class YOLOv5Classifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()

        # Load YOLOv5 CLASSIFICATION model
        self.backbone = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="../yolov5s-cls.pt",  # Path from src/ folder
            trust_repo=True,
        )

        freeze_layers = [
            "model.0.",
            "model.1.",
            "model.2.",
            "model.3.",
            "model.4.",
            "model.5.",
            "model.6.",
            "model.7.",
            "model.8.",
        ]

        for name, param in self.backbone.model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False  # Freeze backbone
            else:
                param.requires_grad = True  # Train head (layer 9)

        # Replace final linear layer: 1000 classes → 102 classes
        # model.9.linear is the classification layer
        in_features = self.backbone.model.model[9].linear.in_features  # 1280
        self.backbone.model.model[9].linear = nn.Linear(in_features, num_classes)

        self.model = self.backbone.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=LEARNING_RATE
        )

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader=None, val_loader=None):

        # 0. Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Store history for plotting later
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        # ============================================
        # TRAINING LOOP
        # ============================================
        for epoch in range(EPOCHS):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0

            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = 100 * train_correct / len(self.train_loader.dataset)
            avg_train_loss = train_loss / len(self.train_loader)

            # Validate
            self.model.eval()
            val_correct = 0
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = self.forward(images)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    # add loss calculation for validation
                    val_loss += self.criterion(outputs, labels).item()

            val_acc = 100 * val_correct / len(self.val_loader.dataset)
            avg_val_loss = val_loss / len(self.val_loader)

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1}/{EPOCHS} - Train: {train_acc:.2f}% - Val: {val_acc:.2f}%"
            )
        return self.history

    def test_model(self, test_loader=None):
        self.test_loader = test_loader
        # ============================================
        # TEST
        # ============================================
        self.model.eval()
        test_correct = 0
        test_loss = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.forward(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_loss += self.criterion(outputs, labels).item()

        test_acc = 100 * test_correct / len(self.test_loader.dataset)
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        test_loss = test_loss / len(self.test_loader)
        return test_acc, test_loss
