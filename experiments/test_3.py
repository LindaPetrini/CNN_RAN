import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, classes):
        super(CNN, self).__init__()
        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(300, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(16*int(self.pad_len/2), self.classes)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = self.sm(out)
        return out



