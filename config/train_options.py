from .options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_model_freq', type=int, default=20000, help='frequency of saving checkpoints')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=500000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for rmsprop in d')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for rmsprop in g')
        parser.add_argument('--loss_type', type=str, default='hinge', choices=['hinge', 'vgan', 'wgan'], help='using type of gan loss')
        parser.add_argument('--num_source_class', type=int, default=119, help='training classes in dataset')
        parser.add_argument('--lam_gp', type=float, default=10.0, help='weight for gradient penalty of gan')
        parser.add_argument('--lam_fp', type=float, default=1.0, help='weight for feature matching')
        parser.add_argument('--lam_recon', type=float, default=0.1, help='weight for recon loss')

        self.isTrain = True
        return parser