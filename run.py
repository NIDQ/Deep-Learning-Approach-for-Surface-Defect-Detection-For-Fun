import argparse
from agent import Agent

DefaultParam = {
    'mode': 'testing',
    'train_mode': 'decision',
    'epochs_num': 50,
    'batch_size': 1,
    'learn_rate': 0.001,
    'momentum': 0.9,
    'data_dir':'dataset/KolektorSDD',
    'checkPoint_dir': 'checkpoint',
    'Log_dir': 'log',
    'valid_ratio': 0,
    'valid_frequency' :3,
    'save_frequenchy': 2,
    'max_to_keep':10,
    'b_restore':True,
    'b_saveNG':True,
}


def parse_arguments():

    parser = argparse.ArgumentParser(description='Train or test the CRNN model.')

    parser.add_argument(
        '--train_segment',
        action='store_true',
        help='Define if we wanna to train the segment net'
    )
    parser.add_argument(
        '--train_decision',
        action='store_true',
        help='Define if we wanna to train the decision net'
    )
    parser.add_argument(
        '--train_total',
        action='store_true',
        help='Define if we wanna to train the total net'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Define if we wanna to train the total net'
    )

    return parser.parse_args()

def main():

    param = DefaultParam
    args = parse_arguments()
    if not args.train_segment and not args.train_decision and not args.train_total and not args.test:
        print('If we are not training, and not testing, what is the point?')
    if args.train_segment:
        param['mode']='training'
        param['train_mode']='segment'
    if args.train_decision:
        param['mode']='training'
        param['train_mode']='decision'
    if args.train_total:
        param['mode']='training'
        param['train_mode']='total'
    if args.train_segment:
        param['mode']='testing'

    param["data_dir"] = args.data_dir
    param["valid_ratio"] = args.valid_ratio
    param["batch_size"] = args.batch_size
    param["epochs_num"] = args.epochs_num
    param["checkPoint_dir"] = args.checkPoint_dir

    agent = Agent(param)
    agent.run()

if __name__ == '__main__':
    main()
