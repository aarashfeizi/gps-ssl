# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.methods.barlow_twins import BarlowTwins
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.mae import MAE
from solo.methods.nnclr import NNCLR
from solo.methods.simclr import SimCLR
from solo.methods.vicreg import VICReg


METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "mae": MAE,
    "nnclr": NNCLR,
    "simclr": SimCLR,
    "vicreg": VICReg,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "MAE",
    "NNCLR",
    "SimCLR",
    "VICReg",
]
