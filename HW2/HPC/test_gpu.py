#!/usr/bin/env python3
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device to use: {device}")
