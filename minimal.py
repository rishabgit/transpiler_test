import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim, nn
import ivy


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

PATH = 'model.pt'
torch.save(model, PATH)

input_tensor = torch.randn(1, 3, 32, 32)
np_image = input_tensor.detach().cpu().numpy()

def inference_wrapper(PATH, input_tensor):
  loaded_model = torch.load(PATH)
  loaded_model.eval()
  return loaded_model(input_tensor)

print(inference_wrapper(PATH, input_tensor).shape)

from transpiler.transpiler import transpile
transpiled_inf_wrap = transpile(inference_wrapper, 
    source='torch', 
    to='jax')

# transpiled_inf_wrap = ivy.transpile(inference_wrapper, 
#     source='torch', 
#     to='jax')

print(transpiled_inf_wrap(PATH, np_image).shape)
