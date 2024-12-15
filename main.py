from auditor_utils import *
import argparse

def cria_pastas(path, is_images=False):
    if not is_images:
        for files in os.listdir(path):
            if not os.path.isdir(os.path.join(path, files)):
                os.system(f"mkdir {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
                os.system(f"mv {os.path.join(path, files)} {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
    else:
        for files in os.listdir(path):
            if os.path.isdir(os.path.join(path, files)):
                os.system(f"mv {os.path.join(path, files)} {os.path.join(path, 'images_orig')}")
                os.system(f"mkdir {os.path.join(path, files)}")
                os.system(f"mv {os.path.join(path, 'images_orig')} {os.path.join(path, files, 'images_orig')}")


def get_video_type(path, dir_file):
    allowed_video_types = ['.mp4', '.MOV', '.mov']
    files = os.listdir(str(os.path.join(path, dir_file)))
    for file in files:
        for video_type in allowed_video_types:
            if file.endswith(video_type):
                return video_type
    return None

def main(path, models, downscale_factor, frames_number, is_images=False, split_fraction=0.9):
    for file in os.listdir(path):
        file_type = get_video_type(path, file)
        if file_type is None:
            print(f'Video not found in dir {file}')
            continue

        output = pipeline(
            path,
            file,
            file + file_type,
            "pilot",
            os.path.join(path, file),
            os.path.join(path, file, "output"),
            models,
            downscale_factor,
            frames_number,
            is_images,
            split_fraction=split_fraction)

        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

parser = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parser.add_argument("-vd", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-id", "--images_dir", type=str, help="Folder with images folders. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-i", "--initialize", type=bool, help="To initialize the videos folder.", default=False)
parser.add_argument("-df", "--downscale_factor", type=int, help="Number of downscale to be used", default=None)
parser.add_argument("-fn", "--frames_number", type=int, help="Number of downscale to be used", default=300)
parser.add_argument("-sf", "--split_fraction", type=float, help="Fraction to divide train/eval dataset", default=0.9)

# Parse arguments
args = parser.parse_args()

models = [
    'nerfacto'
]

if args.initialize:
    if args.videos_dir:
        cria_pastas(args.videos_dir)
    elif args.images_dir:
        cria_pastas(args.images_dir, is_images=True)
else:
    if args.images_dir:
        main(args.images_dir, models, args.downscale_factor, None, is_images=True, split_fraction=args.split_fraction)
    elif args.videos_dir:
        main(args.videos_dir, models, args.downscale_factor, args.frames_number, split_fraction=args.split_fraction)
# main("/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_4", models)
