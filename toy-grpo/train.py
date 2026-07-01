import torch
import torch.nn as nn

from model.policy import TinyPolicy
from grpo.grpo_step import step

def train(model, optmizer, num_steps = 1000):

    print("Toy GRPO on ""addition of two digits (0-9)")

    for steps in range(num_steps):
        acc = step(model, optmizer)
        if steps % 100 == 0:
            print(f"Step: {steps}, acc: {acc:.3f}")