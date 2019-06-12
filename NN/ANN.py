import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import datasets, transforms
from ResNet import ResNet18

BATCH_SIZE = 64
LR = 1e-3
EPOCH = 1000
PRINT_EVERY = 100
BEST_ACCURACY = 0.5
T = 2
alpha = 0.8

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.out(x)
        return y

def train(model, device, train_loader, optimizer, epoch, teacher, criterion):
    model.train()
    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        out_T = teacher(x)
        out_T = F.softmax(out_T/T, dim=1)
        x = x.view(x.size(0), -1)
        output = model(x)
        output = F.log_softmax(output/T, dim=1)
        optimizer.zero_grad()
        loss = (1 - alpha) * F.cross_entropy(output, y) + alpha * criterion(output, out_T)*T*T
        loss.backward()
        optimizer.step()
        if batch % PRINT_EVERY == 0:
            print(
                'EPOCH:{} [{}/{} ({:.0f}%)]\Loss:{:.6f}'.format(
                    epoch, batch * len(x), len(train_loader),
                           100 * batch / len(train_loader), loss.item()
                )
            )

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()
            pred = output.max(1, keepdim=True)[1]
            accuracy += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\n')
    print('Test: Average Loss:{:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, accuracy, len(test_loader),
        100 * accuracy / len(test_loader)
    ))
    global BEST_ACCURACY
    if (100 * accuracy / len(test_loader)) > BEST_ACCURACY:
        BEST_ACCURACY = 100 * accuracy / len(test_loader)
        torch.save(model.state_dict(), 'ANN_byResNet_params.pkl')
    print('BEST TEST ACCURACY:{}\n'.format(BEST_ACCURACY))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = Data.DataLoader(
        datasets.MNIST('data', train=True, download='True',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ])),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = Data.DataLoader(
        datasets.MNIST('data', train=False, download='True',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ]))
    )

    model = ANN().to(device)
    teacher = ResNet18().to(device)
    teacher.load_state_dict(torch.load('ResNet18_params.pkl'))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.KLDivLoss()

    for epoch in range(EPOCH):
        train(model, device, train_loader, optimizer, epoch, teacher, criterion)
        test(model, device, test_loader)

if __name__=="__main__":
    main()
