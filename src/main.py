from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from preprocessing import preprocess_for_vgg19, preprocess_for_yolov5
from train_vgg19 import VGG19Classifier
from train_yolov5 import YOLOv5Classifier
from evaluate import plot_cross_entropy_loss, plot_accuracy_graph


models = ["yolo", "vgg"]
splits = ["split_42", "split_123"]

for model in models:
    for split in splits:

        # loading transform fucntions
        if model == "vgg":
            train_transform, test_transform = preprocess_for_vgg19()
            classifier = VGG19Classifier()
        else:
            train_transform, test_transform = preprocess_for_yolov5()
            classifier = YOLOv5Classifier()

        train_dataset = ImageFolder(
            f"../data/splits/{split}/train", transform=train_transform
        )
        val_dataset = ImageFolder(
            f"../data/splits/{split}/val", transform=test_transform
        )
        test_dataset = ImageFolder(
            f"../data/splits/{split}/test", transform=test_transform
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"Model: {model.upper()}, Split: {split}")

        history = classifier.train_model(
            train_loader=train_loader, val_loader=val_loader
        )
        test_acc, test_loss = classifier.test_model(test_loader=test_loader)

        plot_cross_entropy_loss(
            history, model_name=model, split_name=split, test_loss=test_loss
        )
        plot_accuracy_graph(
            history, model_name=model, split_name=split, test_acc=test_acc
        )
