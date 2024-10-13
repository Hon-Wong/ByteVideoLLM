import re
import torch.nn as nn


def build_projector(
        type: str = "linear", 
        input_hidden_size: int = 1024, 
        output_hidden_size: int = 1024
    ):
    """ build vision projector
    Args:
        type: projector type (linear, mlp2x_gelu, identity)
        input_hidden_size: input hidden size from adaptor
        output_hidden_size: output hidden size to llm
    Returns:
        vision projector module(nn.Module)
    """

    if type == 'linear':
        return nn.Linear(input_hidden_size, output_hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(input_hidden_size, output_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*modules)

    if type == 'identity':
        return nn.Identity()

    raise ValueError(f'Unknown projector type: {type}')
