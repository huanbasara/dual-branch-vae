"""
Simplified SVGTensor class for our dual-branch VAE project.
Contains only the essential definitions needed by config.py.
"""
import torch
from typing import Union

Num = Union[int, float]


class SVGTensor:
    """Simplified SVGTensor class containing only essential constants."""
    
    # Commands used in simplified SVG representation
    COMMANDS_SIMPLIFIED = ["m", "l", "c", "EOS", "SOS"]
    
    # Command argument masks - which arguments are used for each command
    CMD_ARGS_MASK = torch.tensor([[0, 0, 0, 0, 1, 1],   # m (move)
                                  [0, 0, 0, 0, 1, 1],   # l (line)
                                  [1, 1, 1, 1, 1, 1],   # c (cubic bezier)
                                  [0, 0, 0, 0, 0, 0],   # EOS (end of sequence)
                                  [0, 0, 0, 0, 0, 0]])  # SOS (start of sequence)

    class Index:
        """Tensor indexing constants."""
        COMMAND = 0
        START_POS = slice(1, 3)
        CONTROL1 = slice(3, 5)
        CONTROL2 = slice(5, 7)
        END_POS = slice(7, 9)

    class IndexArgs:
        """Argument indexing constants."""
        CONTROL1 = slice(0, 2)
        CONTROL2 = slice(2, 4)
        END_POS = slice(4, 6)

    position_keys = ["control1", "control2", "end_pos"]
    all_position_keys = ["start_pos", *position_keys]
