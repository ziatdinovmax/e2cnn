from e2cnn.nn import GeometricTensor
from e2cnn.nn import FieldType
from .equivariant_module import EquivariantModule
import torch

from typing import List, Tuple, Union, Any

__all__ = ["IdentityModule"]


class IdentityModule(EquivariantModule):
    def __init__(self, in_type):
        r"""

        Simple module which does not perform any operation on the input tensor.

        Args:
            in_type (FieldType): input (and output) type of this module

        """

        super(IdentityModule, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

    def forward(self, input):
        r"""

        Returns the input tensor.

        Args:
            input (GeometricTensor): the input GeometricTensor

        Returns:
            the output tensor

        """

        assert input.type == self.in_type
        return input

    def evaluate_output_shape(self, input_shape):

        assert len(input_shape) > 1
        assert input_shape[1] == self.in_type.size

        return input_shape

    def check_equivariance(self, atol = 2e-6, rtol = 1e-5):
        return super(IdentityModule, self).check_equivariance(atol=atol, rtol=rtol)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Identity` module and set to "eval" mode.

        .. warning ::
            Only working with PyTorch >= 1.2

        """
        self.eval()
        return torch.nn.Identity()
