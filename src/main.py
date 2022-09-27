import yaml,time
import argparse
from scMTG import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.load(f)
    #model = scMTG(params)
    model = scDEC(params)
    model.train()