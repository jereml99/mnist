import glob
from pathlib import Path
import torch

if __name__ == '__main__':

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Normalize(0, 1)
    ])

    raw_path = Path(__file__).parent.parent / "data" / "raw"
    processed_path = Path(__file__).parent.parent / "data" / "processed"

    train_images = [torch.load(train_file) for train_file in glob.glob(str(raw_path / "train_images_*.pt"))]
    train_labels = [torch.load(train_file) for train_file in glob.glob(str(raw_path / "train_target_*.pt"))]

    test_images = torch.load(str(raw_path / "test_images.pt"))
    test_labels = torch.load(str(raw_path / "test_target.pt"))

    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    train_images = transform(train_images)
    test_images = transform(test_images)

    torch.save(train_images, str(processed_path / "train_images.pt"))
    torch.save(train_labels, str(processed_path / "train_target.pt"))
    torch.save(test_images, str(processed_path / "test_images.pt"))
    torch.save(test_labels, str(processed_path / "test_target.pt"))