import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained vo model (default: "")')
    parser.add_argument('--data-root', default='',
                        help='data root dir (default: "")')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='start frame (default: 0)')
    parser.add_argument('--end-frame', type=int, default=-1,
                        help='end frame (default: -1)')
    parser.add_argument('--train-epoch', type=int, default=10,
                    help='number of training epoches (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--print-interval', type=int, default=1,
                        help='the interval for printing the loss (default: 1)')
    parser.add_argument('--snapshot-interval', type=int, default=1000,
                        help='the interval for snapshot results (default: 1000)')
    parser.add_argument('--project-name', default='',
                        help='name of the peoject (default: "")')
    parser.add_argument('--train-name', default='',
                        help='name of the training (default: "")')
    parser.add_argument('--result-dir', default='',
                        help='root directory of results (default: "")')
    parser.add_argument('--save-model-dir', default='',
                        help='root directory for saving models (default: "")')
    parser.add_argument('--loss-weight', default='(1,1,1,1)',
                        help='weight of the loss terms (default: \'(1,1,1,1)\')')
    parser.add_argument('--vo-optimizer', default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='VO optimizer: adam, rmsprop, sgd (default: adam)')
    parser.add_argument('--data-type', default='tartanair', choices=['tartanair', 'kitti', 'euroc'],
                        help='data type: tartanair, kitti, euroc (default: "tartanair")')
    parser.add_argument('--fix-model-parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')
    parser.add_argument('--rot-w', type=float, default=1,
                        help='loss rot part weight (default: 1)')
    parser.add_argument('--trans-w', type=float, default=1,
                        help='loss trans part weight (default: 1)')
    parser.add_argument('--train-portion', type=float, default=1,
                        help='portion to bp loss (default: "False")')
    parser.add_argument('--use-gt-scale', action='store_true', default=False,
                        help='use gt scale to correct trans scale (default: "False")')
    parser.add_argument('--enable-mapping', action='store_true', default=False,
                        help='enable mapping, generate point cloud (default: "False")')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='start epoch (default: 1)')

    args = parser.parse_args()
    args.loss_weight = eval(args.loss_weight)   # string to tuple

    return args
