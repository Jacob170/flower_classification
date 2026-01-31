import os
import shutil
import scipy.io
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = "../data/raw"
LABELS_FILE = "../data/raw/imagelabels.mat"
OUTPUT_DIR = "../data/splits"


def get_images_and_labels():
    """Load all images and labels"""
    # Get image paths
    images = sorted(
        [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".jpg")]
    )

    # Load labels (convert from 1-102 to 0-101)
    labels = scipy.io.loadmat(LABELS_FILE)["labels"][0] - 1

    return images, labels


def create_split(images, labels, seed):
    """Create one train/val/test split"""
    # 50% train, 50% temp
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.5, random_state=seed, stratify=labels
    )

    # 25% val, 25% test
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )

    # Save to folders
    split_dir = f"{OUTPUT_DIR}/split_{seed}"

    for split_name, imgs, lbls in [
        ("train", train_imgs, train_labels),
        ("val", val_imgs, val_labels),
        ("test", test_imgs, test_labels),
    ]:
        for img, label in zip(imgs, lbls):
            # Create class folder
            class_folder = f"{split_dir}/{split_name}/class_{label:03d}"
            os.makedirs(class_folder, exist_ok=True)

            # Copy image
            shutil.copy2(img, class_folder)

    print(
        f"Split {seed}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}"
    )


# Main
if __name__ == "__main__":
    images, labels = get_images_and_labels()
    print(f"Total images: {len(images)}, Classes: {len(set(labels))}")

    # Create 2 splits
    create_split(images, labels, seed=42)
    create_split(images, labels, seed=123)

    print(f"\nDone! Splits saved in {OUTPUT_DIR}")
