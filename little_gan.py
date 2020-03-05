import torch
import matplotlib.pyplot as plt
import numpy as np


# Target distribution samples Gamma distributed
gamma = torch.distributions.Gamma(1, 3)
X_true = gamma.sample((1000, 1))

# Target distribution samples MoG distributed (Exhibits mode collapse)
# X_0 = torch.randn(1000)
# X_true = torch.cat([torch.randn(500)*0.33 + 1.5, torch.randn(500)*0.16 - 0.5]).reshape(-1,1)

# Simple discriminator class
class Discriminator(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden_1, n_outputs):
        super(Discriminator, self).__init__()
        self.L1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.L2 = torch.nn.Linear(n_hidden_1, n_outputs)

    def forward(self, x):
        h_1 = torch.tanh(self.L1(x))
        y = torch.sigmoid(self.L2(h_1))
        return y


# Simple generator class
class Generator(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden_1, n_outputs):
        super(Generator, self).__init__()
        self.L1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.L2 = torch.nn.Linear(n_hidden_1, n_outputs)

    def forward(self, z):
        h_1 = torch.tanh(self.L1(z))
        y = self.L2(h_1)
        return y


# Instantiate discriminator and generators
d = Discriminator(1, 16, 1)
g = Generator(1, 16, 1)
Z_0 = torch.randn(1000, 1)

# Run for 25000 iterations
n_iters = 2500

# Number of samples per update step
m = 64

# Number of discriminator updates per generator update
k = 1

# Separate optimizers for discriminator and generator
optimizer_disc = torch.optim.RMSprop(d.parameters(), lr=1e-2, weight_decay=0)
optimizer_gen = torch.optim.RMSprop(g.parameters(), lr=1e-4, weight_decay=1e-3)

# Animate the training process
plt.ion()
fig, axs = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(hspace=0)
bins = np.linspace(-1, 3, 51)
a_true = np.argsort(X_true.detach().numpy().ravel())

# Loop over n_iters epochs
for i in range(n_iters):
    # Train the discriminator
    for j in range(k):
        optimizer_disc.zero_grad()
        optimizer_gen.zero_grad()

        # Draw a random sample from the observations
        X_sample = X_true[torch.randperm(len(X_true))[:m]]

        # Draw a random sample from the latent variables, and run through
        # the generator to produce "fake" X
        z = torch.randn(m, 1)
        X_fake = g(z)

        # Run the discriminator on both true and fake samples
        D_true = d(X_sample)
        D_fake = d(X_fake)

        # Compute BCE loss for discriminator
        cost_disc = -torch.mean(
            torch.log(D_true) + torch.log(1 - D_fake)
        )  # Goodfellow Eq. 1

        # Update discriminator parameters
        cost_disc.backward()
        optimizer_disc.step()

    # Train the generator
    optimizer_disc.zero_grad()
    optimizer_gen.zero_grad()

    # Draw a random sample from the latent variables, and run through the
    # generator to produce "fake" X
    z = torch.randn(m, 1)
    X_fake = g(z)

    # Run through the discriminator
    D_fake = d(X_fake)

    # Update generator parameters so that it does a better job fooling
    # the discriminator
    cost_gen = torch.mean(
        -torch.log(D_fake)
    )  # Per Goodfellow, this does better than log(1 - D_fake)

    # Update generator parameters
    cost_gen.backward()
    optimizer_gen.step()
    print(cost_disc.item(), cost_gen.item())
    if i % 50 == 0:
        ax = axs[0]
        ax.cla()

        # Create fake samples for plotting
        X_f = g(Z_0)
        # Evaluate descriminator over all X in a reasonable domain
        dd = d(torch.tensor(bins, dtype=torch.float).reshape(-1, 1))

        # Plot histogram of true samples
        ax.hist(
            X_true.detach().numpy().ravel(),
            bins,
            density=True,
            histtype="step",
            linestyle=(0, (1, 2)),
            linewidth=2.0,
            color="black",
        )
        # Plot histogram of fake samples
        ax.hist(
            X_f.detach().numpy().ravel(),
            bins,
            density=True,
            histtype="step",
            linestyle="solid",
            color="green",
        )
        # Plot discriminator probability as a function of X
        ax.plot(bins, dd.detach().numpy().ravel(), "b:")
        ax.set_xlim(bins.min(), bins.max())
        ax.set_ylim(0, 2.5)
        ax.set_xticks([])
        ax.set_ylabel("P(X)")

        ax = axs[1]
        ax.cla()
        [
            plt.plot([z, x], [0.0, 1.0], "k-")
            for z, x in zip(Z_0[::100], X_f[::100])
        ]
        ax.set_xlim(bins.min(), bins.max())
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Z(bottom) - X(top)")

        plt.pause(1e-10)
