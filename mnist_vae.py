import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z


class Gaussian(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(Gaussian, self).__init__()

        self.num_classes = num_classes
        self.mean = nn.Parameter(torch.zeros(self.num_classes, latent_dim), requires_grad=True) # 5， 128

    def forward(self, z): # 16， 128
        z = z.unsqueeze(1) # 16， 1， 128
        return z - self.mean.unsqueeze(0) # 16， 5， 128


class classifier(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return F.softmax(self.fc2(h)) # 16， 5


class VAEClusterModel(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, num_classes):
        super(VAEClusterModel, self).__init__()
        self.z_dim = z_dim
        self.model = VAE(x_dim, h_dim1, h_dim2, z_dim)
        self.gaussian = Gaussian(num_classes, z_dim)
        self.classifier = classifier(num_classes, z_dim)

    def forward(self, inputs):
        x_recon, z_mean, z_log_var, z = self.model(inputs)
        z_prior_mean = self.gaussian(z)
        y = self.classifier(z)

        return x_recon, z_mean, z_log_var, z, z_prior_mean, y

    def generate(self, y):
        out = []
        for i in range(0, 64):
            z = torch.randn(1, self.z_dim) + self.gaussian.mean[y].unsqueeze(0)
            sample = self.model.decoder(z).view(1, 28, 28)
            out.append(sample)
        out = torch.stack(out, dim=0)
        return out


def loss_function(recon_x, x, log_var, z_prior_mean, y):

    lamb = 2.5
    #xent_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    xent_loss = 0.5 * torch.mean((x.view(-1, 784) - recon_x)**2, 0)
    # print(xent_loss)
    kl_loss = -0.5 * (log_var.unsqueeze(1) - torch.square(z_prior_mean))
    kl_loss = torch.mean(torch.matmul(y.unsqueeze(1), kl_loss), 0)
    cat_loss = torch.mean(y * torch.log(y + 1e-10), 0)
    vae_loss = lamb * torch.sum(xent_loss) + torch.sum(kl_loss) + torch.sum(cat_loss)
    return vae_loss


# def loss_function(recon_x, x, mu, log_var):
#
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#
#     return BCE + KLD


if __name__ == '__main__':
    # model = VAE(nz=20)
    # inputs = torch.randn((1,1,28,28))
    # out = model(inputs)
    # build model
    model = VAEClusterModel(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=20, num_classes=10)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)


    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data.cuda()
            optimizer.zero_grad()

            recon_batch, mu, log_var, z, z_prior_mean, y = model(data)
            # print(recon_batch.shape)
            #loss = loss_function(recon_batch, data, log_var, z_prior_mean, y)
            loss = loss_function(recon_batch, data, log_var, z_prior_mean, y)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


    def test():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                recon, mu, log_var, z, z_prior_mean, y = model(data)

                # sum up batch loss
                test_loss += loss_function(recon, data, log_var, z_prior_mean, y).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    for epoch in range(1, 101):
        train(epoch)
        test()
        torch.save(model.state_dict(), 'best_model.pth')

    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()
    for i in range(0, 10):
        out = model.generate(i)
        save_image(out, 'test_{:}.png'.format(i))

