

import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.current_device())  # Check the current device index
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Check the GPU name
# adding to git
print(torch.cuda.get_device_capability())

print('-----')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.cuda.get_device_properties(device))
print('-----')

#%%
x = torch.tensor([1.0, 2.0, 3.0], device=device)
y = torch.tensor([4.0, 5.0, 6.0], device=device)
result = x + y

print(result)



print(torch.cuda.is_available())

# setting device on GPU if available, else CPU

print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
