import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')

# training hyper-parameters
parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')

# model hyper-parameters
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nclass', type=int, default=2)

# misc
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model_path', default='', help="path to saved model (to continue training)")
parser.add_argument('--outf', default=None, help='folder to output images and model checkpoints')
parser.add_argument('--logdir', default="./logdir", help='folder to log dir')
parser.add_argument('--log_step', type=int, default=5)
parser.add_argument('--sample_epoch', type=int, default=2)

def get_config():
    return parser.parse_args()
