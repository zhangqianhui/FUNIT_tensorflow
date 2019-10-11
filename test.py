from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from Dataset import Flowers
from FUNIT import FSUGAN
from config.test_options import TestOptions

opt = TestOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES']= str(opt.gpu_id)

if __name__ == "__main__":

    d_ob = Flowers(opt)
    dwgan = FSUGAN(d_ob, opt)
    dwgan.build_model()

    dwgan.test()

