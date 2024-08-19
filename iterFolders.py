from os import listdir
import sys
import argparse

def main(args):
    weights_file = args.weightsFile
    rootPath = args.rootPath
    height = args.height
    width = args.width
    numberFolders = args.numberFolders
    
    model = load_model(weights_file)
    
    evaluateFolders(model, rootPath, height, width, numberFolders)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weightsFile', type=str, help='Saved CNN model file', default='./checkpoint/cp.h5')
    
    parser.add_argument('--rootPath', type=str, help='Directory with (Moiré pattern) images.', default='./')
    
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    parser.add_argument('--numberFolders', type=int, help='Number of folders', default=0)
        
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))