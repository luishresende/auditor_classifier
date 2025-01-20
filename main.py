from auditor_utils import *
import argparse

class Properties:
    def add_property(self, name, value):
        setattr(self, name, value)

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

def render(path, models):
    for file in os.listdir(path):
        for model in models:
            os.system(
                'ns-render interpolate ' +
                f'--load-config {os.path.join(path, file, "output_" + model, file, "*", "*", "config.yml")} ' +
                f'--output-path {os.path.join(path, file, "output_" + model, file + "_" + model + "_1.mp4")} ' + 
                '--downscale-factor 0.25 ' + 
                '--frame-rate 30 ' + 
                '--interpolation-steps 60'
            )

def main(path, models, is_images=False, propert=None):
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
            is_images,
            propert=propert)

        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

parser = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parser.add_argument("-vd", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-id", "--images_dir", type=str, help="Folder with images folders. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-i", "--initialize", type=bool, help="To initialize the videos folder.", default=False)
parser.add_argument("-r", "--render", type=bool, help="Wether to render videos from models.", default=False)
parser.add_argument("-df", "--downscale_factor", type=int, help="Number of downscale to be used", default=None)
parser.add_argument("-fn", "--frames_number", type=int, help="Number of downscale to be used", default=300)
parser.add_argument("-sf", "--split_fraction", type=float, help="Fraction to divide train/eval dataset", default=0.9)
parser.add_argument("-mi", "--max_num_iterations", type=int, help="Maximum number of iterations during training", default=50000)

# Parse arguments
args = parser.parse_args()

# Add properties
propert = Properties()
propert.add_property('downscale_factor', args.downscale_factor)
propert.add_property('frames_number', args.frames_number)
propert.add_property('split_fraction', args.split_fraction)
propert.add_property('max_num_iterations', args.max_num_iterations)


models = [
    'nerfacto'
]

if args.initialize:
    if args.videos_dir:
        cria_pastas(args.videos_dir)
    elif args.images_dir:
        cria_pastas(args.images_dir, is_images=True)
else:
    if not args.render:
        if args.images_dir:
            main(args.images_dir, models, is_images=True, propert=propert)
        elif args.videos_dir:
            main(args.videos_dir, models, propert=propert)
    else:
        if args.images_dir:
            render(args.images_dir, models)
        elif args.videos_dir:
            render(args.videos_dir, models)
