import argparse
import os
from PyLib.utils import makefolders

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--data_dir', type=str, default='/home/dataset/flowers/', help='path to images')
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--image_size', type=int, default=128, help='scale images to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--necf_t', type=int, default=64, help='# of translation encode content filters in first conv layer')
        parser.add_argument('--nesf_t', type=int, default=64, help='# of translation encode style filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of d filters in first conv layer')
        parser.add_argument('--n_g_ref_t', type=int, default=512, help='# of generator filters in residual block')
        parser.add_argument('--n_layers_ec', type=int, default=3, help='layers of encode content model')
        parser.add_argument('--n_layers_es', type=int, default=4, help='layers of encode style model')
        parser.add_argument('--n_layers_de', type=int, default=3, help='layers of decoder model')
        parser.add_argument('--n_layers_D', type=int, default=4, help='layers of d model')
        parser.add_argument('--n_residual_de', type=int, default=2, help='number of residual of decoder')
        parser.add_argument('--n_residual_d', type=int, default=5, help='number of residual of d')
        parser.add_argument('--gpu_id', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--exper_name', type=str, default='log9_20_5', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs for tensorboard')
        parser.add_argument('--sample_dir', type=str, default='./sample_dir', help='dir for sample images')
        parser.add_argument('--test_sample_dir', type=str, default='sample_img', help='test sample images are saved hear')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.initialized = True
        return parser

    def gather_options(self):

        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        # save to the disk
        file_name = os.path.join(opt.checkpoints_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        opt.checkpoints_dir = os.path.join(opt.exper_name, opt.checkpoints_dir)
        opt.sample_dir = os.path.join(opt.exper_name, opt.sample_dir)
        opt.log_dir = os.path.join(opt.exper_name, opt.log_dir)
        makefolders([opt.checkpoints_dir, opt.sample_dir, opt.log_dir])

        self.print_options(opt)
        self.opt = opt

        return self.opt
