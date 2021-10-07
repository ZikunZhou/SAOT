# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# OTB2015
__C.WINDOW_INFLUENCE = 0.15023374544821316
__C.PENALTY_K = 0.4670975808285926
__C.LR = 0.7515981966269943


# VOT2018
#__C.WINDOW_INFLUENCE = 0.29668172697835943
#__C.PENALTY_K = 0.16150775431376707
#__C.LR = 0.14481461047031524


# NFS30
#__C.WINDOW_INFLUENCE = 0.6394239201925985
#__C.PENALTY_K = 0.12418040866085275
#__C.LR = 0.7830185033454389