import torch
import torchreid
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
)
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

def finetune_reid_model(dataset_path: Path, output_dir: Path, epochs: int):
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize((256, 128)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=str(dataset_path), transform=transform)
    
    if not train_dataset.classes:
         raise ValueError("No class folders found in the dataset directory.")

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=num_classes,
        loss='triplet',
        pretrained=True
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.0003)
    scheduler = StepLR(optimizer, step_size=int(epochs * 0.7), gamma=0.1)
    
    loss_fn = torch.nn.TripletMarginLoss(margin=0.3)

    for epoch in range(epochs):
        model.train()
        
        for images, pids in train_loader:
            images, pids = images.to(device), pids.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            embeddings = output[0] if isinstance(output, (tuple, list)) else output
            
            anchor_idx, positive_idx, negative_idx = [], [], []
            for i in range(images.size(0)):
                pos_mask = (pids == pids[i]) & (torch.arange(images.size(0)).to(device) != i)
                neg_mask = pids != pids[i]

                if torch.any(pos_mask) and torch.any(neg_mask):
                    pos_i = torch.where(pos_mask)[0]
                    neg_i = torch.where(neg_mask)[0]
                    
                    anchor_idx.append(i)
                    positive_idx.append(pos_i[torch.randint(len(pos_i), (1,))].item())
                    negative_idx.append(neg_i[torch.randint(len(neg_i), (1,))].item())

            if not anchor_idx:
                continue
                
            anchor = embeddings[anchor_idx]
            positive = embeddings[positive_idx]
            negative = embeddings[negative_idx]

            loss = loss_fn(anchor, positive, negative)
            
            loss.backward()
            optimizer.step()
            
        scheduler.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / "osnet_x1_0_finetuned.pth"
    torch.save(model.state_dict(), final_model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default="data/reid_dataset_final/bounding_box_train")
    parser.add_argument("--output_dir", type=Path, default="weights/reid_finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    finetune_reid_model(args.dataset_path, args.output_dir, args.epochs)
    
    print("Script finished.")

if __name__ == "__main__":
    main()