import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from ANN import ANN

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANN().to(device)
model.load_state_dict(torch.load('ANN_byResNet_params.pkl'))

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download='True',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ), (0.5, ))
                       ]))
    )

test(model, device, test_loader)
