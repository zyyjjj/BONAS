import logging
import os
from settings import local_root_dir, search_config, logfile, training_config, taskname, results_dir
from BO_tools.runner import Runner
import argparse
parser = argparse.ArgumentParser("search")
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu')
# parser.add_argument('--regression_type', type = str, default = 'linear',
#                     help='specify linear or quantile regression')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logging.basicConfig(filename=os.path.join(local_root_dir, results_dir, taskname, str(args.gpu)+'_'+logfile), filemode='w', level=logging.INFO,
                    format='%(asctime)s : %(levelname)s  %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S')
os.environ["PYTHONHASHSEED"] = "0"
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s  %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# TODO: enable saving log with identifiable names for regression type

if __name__ == "__main__":
    runner = Runner(**search_config, training_cfg=training_config)
    runner.run()

    # 
