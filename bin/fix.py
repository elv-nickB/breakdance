import argparse
import os

def main():
    parser = argparse.ArgumentParser(description = 'Fix Paths')
    parser.add_argument('--dir', type = str, help = "Input file path")
    args = parser.parse_args()
    
    for part in os.listdir(args.dir):
        filepath = os.path.join(args.dir, part)
        renamed = f'0{part[0:4]}.mp4'
        renamed_path = os.path.join(args.dir, renamed)
        os.rename(filepath, renamed_path)


if __name__ == '__main__':
    main()