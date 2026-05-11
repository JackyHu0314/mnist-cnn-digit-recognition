import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def predict(img_path, invert=True):
    img = Image.open(img_path).convert('L')
    if invert:
        img = ImageOps.invert(img)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
    print(f'\n图片: {img_path}')
    print(f'预测结果: {pred}  (置信度 {probs[pred].item()*100:.2f}%)')
    print('各数字概率:')
    for i, p in enumerate(probs):
        bar = '#' * int(p.item() * 30)
        print(f'  {i}: {p.item()*100:5.2f}%  {bar}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python predict.py <图片路径> [图片路径2 ...]')
        print('例如: python predict.py my_digit.png')
        sys.exit(1)
    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f'找不到文件: {path}')
            continue
        predict(path)
