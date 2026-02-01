import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 102
EPOCHS = 20
LEARNING_RATE = 0.001


class VGG19Classifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()

        # 1. Load pretrained backbone
        self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # 2. FREEZE backbone (features only)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        self.in_features = self.backbone.classifier[6].in_features  # = 4096

        self.backbone.classifier[6] = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.model = self.backbone.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=LEARNING_RATE
        )

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
                outputs = self.model(images)
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
                    outputs = self.model(images)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_loss += self.criterion(outputs, labels).item()  # ADD THIS

            val_acc = 100 * val_correct / len(self.val_loader.dataset)
            avg_val_loss = val_loss / len(self.val_loader)  # ADD THIS

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_loss"].append(avg_val_loss)  # ADD THIS

            print(
                f"Epoch {epoch+1}/{EPOCHS} - Train: {train_acc:.2f}% - Val: {val_acc:.2f}%"
            )
        return self.history

    def test_model(self, test_loader=None):
        self.test_loader = test_loader
        self.model.eval()
        test_correct = 0
        test_loss = 0  # ADD

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                test_correct += (outputs.argmax(1) == labels).sum().item()
                test_loss += self.criterion(outputs, labels).item()  # ADD

        test_acc = 100 * test_correct / len(self.test_loader.dataset)
        avg_test_loss = test_loss / len(self.test_loader)  # ADD

        print(f"\nTest Accuracy: {test_acc:.2f}%")
        return test_acc, avg_test_loss  # RETURN BOTH
