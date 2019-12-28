from model import Generator, Discriminator, DHead, QHead
from utils import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

def get_mnist_dataloader():
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor()])
    dataset = dsets.MNIST('mnist/', train='train', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloader

def get_networks():
    netG = Generator().to(device)
    netG.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    netD = DHead().to(device)
    netD.apply(weights_init)

    netQ = QHead().to(device)
    netQ.apply(weights_init)
    return netG, discriminator, netD, netQ

def get_criterion():
    # Loss for discrimination between real and fake images.
    criterionD = nn.BCELoss()
    # Loss for discrete latent code.
    criterionQ_dis = nn.CrossEntropyLoss()
    # Loss for continuous latent code.
    criterionQ_con = NormalNLLLoss()
    return criterionD, criterionQ_dis, criterionQ_con


def main():
    # Dataloader
    dataloader = get_mnist_dataloader()

    # Initialise the networks
    netG, discriminator, netD, netQ = get_networks()

    # Criterion
    criterionD, criterionQ_dis, criterionQ_con = get_criterion()

    # Optimizer
    optimD = torch.optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=1e-3)
    optimG = torch.optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=1e-3)

    # Training loop

    real_label = 1
    fake_label = 0

    G_losses = []
    D_losses = []

    epochs = 2

    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader, 0):
            step_G_loss = []
            step_D_loss = []
            b_size = data.size(0)
            real_data = data.to(device)

        # Discriminator Step
            optimD.zero_grad()

            # Real Data
            label = torch.full((b_size, ), real_label, device=device)
            output1 = discriminator(real_data)
            probs_real = netD(output1).view(-1)
            loss_real = criterionD(probs_real, label)
            loss_real.backward()

            # Generated Data
            label.fill_(fake_label)
            noise, idx = noise_sample(1, 10, 2, 62, b_size, device)
            fake_data = netG(noise)
            output2 = discriminator(fake_data.detach())
            probs_fake = netD(output2).view(-1)
            loss_fake = criterionD(probs_fake, label)
            loss_fake.backward()

            # D loss
            D_loss = loss_real + loss_fake
            optimD.step()


        # Generator Step
            optimG.zero_grad()

            # Generated Data as Real
            output = discriminator(fake_data)
            label.fill_(real_label)
            probs_fake = netD(output).view(-1)
            gen_loss = criterionD(probs_fake, label)

            q_logits, q_mu, q_var = netQ(output)
            target = torch.LongTensor(idx).to(device)
            # Calculating loss for discrete latent code.
            dis_loss = 0
            dis_loss = criterionQ_dis(q_logits[:, :10], target[0])

            con_loss = 0
            con_loss = criterionQ_con(noise[:, 62+ 10 : ].view(-1, 2), q_mu, q_var)*0.1


            # Net loss for generator.
            G_loss = gen_loss + dis_loss + con_loss
            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            optimG.step()

            step_D_loss.append(G_loss.item())
            step_G_loss.append(D_loss.item())

        G_losses.append(sum(step_G_loss)/len(step_G_loss))
        D_losses.append(sum(step_D_loss)/len(step_D_loss))

        # print losses
        print('Epochs:{}\tG Loss: {:.4f}\tD Loss: {:.4f}'.format(epoch+1, G_losses[-1], D_losses[-1]))


    # Plot the training losses.
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    torch.save({
    'netG' : netG.state_dict(),
    'discriminator' : discriminator.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict()
    }, 'trained_infogan.pt')

if __name__ == "__main__":
    main()
