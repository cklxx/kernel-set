from {{PROJECT_NAME_LOWER}} import example_megakernel
import torch

instruction = torch.zeros(148, 1, 32, dtype=torch.int32, device="cuda")
timing = torch.zeros(148, 1, 128, dtype=torch.int32, device="cuda")
instruction[0, 0, 0] = 1

print('Starting example megakernel')
example_megakernel(instruction, timing)