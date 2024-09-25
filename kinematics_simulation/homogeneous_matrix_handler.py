import numpy as np
import torch
"""Include basic function to work with Homogeneous Matrix in numpy"""
"""No unit test for this"""

def identity():
    """No movement"""
    return torch.eye(4)
def translate(x = 0, y = 0, z = 0):
    """Tranlate in x y z direction"""
    x = num_to_tensor(x)
    y = num_to_tensor(y)
    z = num_to_tensor(z)
    translation = torch.stack([x, y, z, torch.tensor(1.0)])
    result = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = torch.concat([result, translation.view(-1, 1)], dim=1)
    return result
def rotate(angle = 0, axis = [0, 0, 1]):
    """Rotate around axis an angle radian"""
    axis = vector_to_tensor(axis)
    axis = axis/axis.norm()
    angle = num_to_tensor(angle)
    result = skew_matrix(axis)
    # Rodrigues' rotation formula:
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    result = torch.eye(3) + torch.sin(angle)*result + (1-torch.cos(angle))*(result@result)
    result = torch.concat([result,torch.tensor([[0],[0],[0]])], dim = 1)
    result = torch.concat([result,torch.tensor([[0, 0, 0, 1]])], dim = 0)
    return result


def num_to_tensor(x):
    if type(x) == torch.Tensor:
        return x
    return torch.tensor(x, dtype=torch.float32)
def vector_to_tensor(x):
    if type(x) == torch.Tensor:
        return x
    return torch.tensor(x, dtype=torch.float32)
def skew_matrix(vector):
    x, y, z = vector.unbind(dim=-1)
    # Create Ot as a tensor with zero
    Ot = num_to_tensor(0.0)
    result = torch.stack([torch.stack([Ot, -z,  y]),
                          torch.stack([ z, Ot, -x]),
                          torch.stack([-y,  x, Ot])])
    return result



