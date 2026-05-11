import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps

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
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

CANVAS = 280
BRUSH = 18

class App:
    def __init__(self, root):
        self.root = root
        root.title('手写数字识别')

        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg='white', cursor='pencil')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', lambda e: self.predict())

        tk.Button(root, text='识别', width=10, command=self.predict).grid(row=1, column=0, pady=5)
        tk.Button(root, text='清空', width=10, command=self.clear).grid(row=1, column=1, pady=5)
        tk.Button(root, text='退出', width=10, command=root.quit).grid(row=1, column=2, pady=5)

        self.result = tk.Label(root, text='在画板上写一个数字 (0-9)', font=('Arial', 16))
        self.result.grid(row=2, column=0, columnspan=3, pady=10)

        self.probs_label = tk.Label(root, text='', font=('Courier', 10), justify='left')
        self.probs_label.grid(row=3, column=0, columnspan=3, pady=5)

        self.image = Image.new('L', (CANVAS, CANVAS), 'white')
        self.draw_img = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH, fill='black', outline='black')
        self.draw_img.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill='black')

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS, CANVAS), 'white')
        self.draw_img = ImageDraw.Draw(self.image)
        self.result.config(text='在画板上写一个数字 (0-9)')
        self.probs_label.config(text='')

    def predict(self):
        img = ImageOps.invert(self.image)
        bbox = img.getbbox()
        if bbox is None:
            self.result.config(text='画板是空的')
            return
        img = img.crop(bbox)
        w, h = img.size
        side = max(w, h) + 40
        square = Image.new('L', (side, side), 'black')
        square.paste(img, ((side - w) // 2, (side - h) // 2))

        x = transform(square).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            pred = probs.argmax().item()

        self.result.config(text=f'预测: {pred}   置信度: {probs[pred].item()*100:.1f}%')
        lines = []
        for i, p in enumerate(probs):
            bar = '#' * int(p.item() * 25)
            lines.append(f'{i}: {p.item()*100:5.1f}%  {bar}')
        self.probs_label.config(text='\n'.join(lines))

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
