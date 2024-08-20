from iterDir import evaluateFolders
from moveImgs import txtGetFiles
import sys
import argparse

def main(args):
    dirPath = args.dirPath
    
    evaluateFolders(dirPath)
    txtGetFiles(dirPath)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--dirPath', type=str, help='Directory with (Moiré pattern) images.', default='./')
        
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))