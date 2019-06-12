import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms
import argparse
from ResNet import ResNet18
from ANN import ANN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
# parser = argparse.ArgumentParser('--out', default='./model/', help='folder to output images and model checkpoints')
# parser = argparse.ArgumentParser('--load', default='./model/ResNet18.pkl', help='load path')
# args = parser.parse_args()

EPOCH = 200
BATCH_SIZE = 64
BEST_ACCURACY = 0.5
LR = 1e-3
PRINT_EVERY = 100

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # x = x.view(x.size(0), -1)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
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
            # x = x.view(x.size(0), -1)
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
    print("BEST:", BEST_ACCURACY)
    if (100 * accuracy / len(test_loader)) > BEST_ACCURACY:
        BEST_ACCURACY = 100 * accuracy / len(test_loader)
        torch.save(model.state_dict(), 'ResNet18_params.pkl')
    print('BEST TEST ACCURACY:{}\n'.format(BEST_ACCURACY))

def main():
    train_loader = Data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.1631, ))
                       ])),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = Data.DataLoader(
        datasets.MNIST('data', train=False
                       , download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.1631, ))
                       ]))
    )
    model = ResNet18().to(device)
    # model = ANN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    BEST_ACCURACY = 0.5

    for epoch in range(1, EPOCH+1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__=="__main__":
    main()