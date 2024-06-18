import os
import sys
import glob
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# 데이터셋 정의
class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augmentation_factor=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation_factor = augmentation_factor
        self.image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
        self.mask_paths = glob.glob(os.path.join(mask_dir, "*.tif"))

        # augmentation_factor 만큼 데이터를 늘림
        self.image_paths *= augmentation_factor
        self.mask_paths *= augmentation_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_conv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv_block(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.functional.max_pool2d(e1, 2))
        e3 = self.encoder3(nn.functional.max_pool2d(e2, 2))
        e4 = self.encoder4(nn.functional.max_pool2d(e3, 2))

        b = self.bottleneck(nn.functional.max_pool2d(e4, 2))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.final_conv(d1)
        return out


# 이미지와 마스크를 256x256 패치로 분할하고 데이터로더를 생성하는 함수
def create_dataloader(image_dir, mask_dir, batch_size, augmentation_factor=1):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ]
    )
    dataset = CellSegmentationDataset(
        image_dir, mask_dir, transform, augmentation_factor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


# 모델 학습 함수
def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    checkpoint_path,
    log_dir,
):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

        scheduler.step()

        # 체크포인트 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}, loss: {best_loss:.4f}")

    writer.close()


# 모델 추론 함수
def infer_model(model, dataloader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    outputs = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            preds = model(images)
            outputs.append(preds.cpu())

    return outputs


if __name__ == "__main__":
    # 주요 경로 설정
    image_dir = "/mnt/c/Users/user/Desktop/Unet/DIC_형광/Img"
    mask_dir = "/mnt/c/Users/user/Desktop/Unet/DIC_형광/Mask"
    checkpoint_path = "unet_checkpoint.pth"
    log_dir = "runs/unet_experiment"

    # 하이퍼파라미터 설정
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델, 손실 함수, 옵티마이저, 스케줄러 설정
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 데이터로더 생성
    dataloader = create_dataloader(
        image_dir, mask_dir, batch_size, augmentation_factor=1
    )

    if sys.argv[1] == "train":
        # 모델 학습
        train_model(
            model,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            checkpoint_path,
            log_dir,
        )
    if sys.argv[1] == "infer":
        # 모델 추론 예시
        outputs = infer_model(model, dataloader, checkpoint_path)
