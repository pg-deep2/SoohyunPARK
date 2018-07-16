import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', default='C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData', help='path to dataset')

parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--log_interval', type=int, default=100, help='save valid gif and image')
parser.add_argument('--checkpoint_step', type=int, default=10, help='save checkpoint')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay for adam. default=0.00001')


parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default=None, help='folder to output images and videos ans model checkpoints')
parser.add_argument('--resnet_path', default="'C:\\Users\msi\Downloads\c3d.pickle'", help='path to pretrained resnet model')
def get_config():
    return parser.parse_args()
