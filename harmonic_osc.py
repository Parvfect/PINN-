"""
Adapted from tutorial at ->
https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb

Possible extensions 
 - Find ideal mix of loss function to reduce number of training steps while fitting to data
 - Add a test accuracy
 - Generalise for other physics problems
"""

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def save_gif(outfile, files, fps=5, loop=0):
    """ Helper function for saving gifs """
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format = 'GIF', append_images=imgs[1:], save_all = True, duration = (1000/fps), loop=loop)


def oscillator(d, wo, x):
    """ Defines analytical solution for 1D Harmonic Oscillator """
    
    assert d < wo
    w = np.sqrt(wo**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y


class FCN(nn.Module):
    """ Defines a fully connected network """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        ])

        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])

        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)


    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# Generating training data

d, w0 = 2, 20 

x = torch.linspace(0,1,500).view(-1,1)
y = oscillator(d, w0, x).view(-1,1)
print(x.shape, y.shape)

# Slice out a small number of points from the LHS of the domain
x_data = x[0:400:20]
y_data = y[0:400:20]
print(x_data.shape, y_data.shape)

"""
plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()
plt.show()
"""

# Training Standard Normal neural network 

def plot_result(x, y, x_data, y_data, yh, training_step, xp=None):
    "Pretty plot training results"
    i = training_step
    plt.figure(figsize=(8,4))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")
    

def train_fcc():
    torch.manual_seed(123)
    model = FCN(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    files = []

    for i in range(800):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh - y_data)**2) # Using mean squared error loss
        loss.backward()
        optimizer.step()

        # Plot training progress
        if (i+1) % 10 == 0:
            yh = model(x).detach()
            plot_result(x, y, x_data, y_data, yh, i)
            file = "plots/nn_%.8i.png"%(i+1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
        
            if (i+1) % 398 == 0: plt.show()
            else: plt.close("all")


    return files

# Physics Informed Neural Network
# Adding a loss function that makes the learning consistent with the differential equation

#Here we evaluate the physics loss at 30 points uniformly spaced over the problem domain .
#We can calculate the derivatives of the network solution with respect to its input variable
# at these points using pytorch's autodifferentiation features, 
#and can then easily compute the residual of the differential equation using these quantities.


def train_pinn():
    x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)# sample locations over the problem domain
    mu, k = 2*d, w0**2


    torch.manual_seed(123)
    model = FCN(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    files = []

    for i in range(12000):
        optimizer.zero_grad()
        
        # compute the "data loss"
        yh = model(x_data)
        loss1 = torch.mean((yh-y_data)**2)# use mean squared error
        
        # compute the "physics loss"
        yhp = model(x_physics)
        dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
        dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
        physics = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
        loss2 = (1e-4)*torch.mean(physics**2)
        
        # Ones like returns array of 1s with the same size as input array
        # torch.autograd.grad -> Computes and returns the sum of gradients of outputs with respect to the inputs.

        # backpropagate joint loss
        loss = loss1 + loss2 # add two loss terms together
        loss.backward()
        optimizer.step()
        
        
        # plot the result as training progresses
        if (i+1) % 150 == 0: 
            
            yh = model(x).detach()
            xp = x_physics.detach()
            
            plot_result(x,y,x_data,y_data,yh,i,xp)
            
            file = "plots/pinn_%.8i.png"%(i+1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            
            if (i+1) % 6000 == 0: plt.show()
            else: plt.close("all")
            
    return files

if __name__ == "__main__":    

    #files = train_fcc()
    #save_gif("pinn.gif", files, fps=20, loop=0)
    files = train_pinn()
    save_gif("pinn.gif", files, fps=20, loop=0)

