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

def main(path, models, is_images=False):
    for file in os.listdir(path):
        output = pipeline(
            path,
            file,
            file + ".mp4" if file.endswith(".mp4") else file + ".MOV",
            "pilot",
            os.path.join(path, file),
            os.path.join(path, file, "output"),
            models,
            is_images)

        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

parser = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parser.add_argument("-vd", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-id", "--images_dir", type=str, help="Folder with images folders. Do not use ./ to refer to the folder. Use the absolute path.", default=None)
parser.add_argument("-i", "--initialize", type=bool, help="To initialize the videos folder.", default=False)

# Parse arguments
args = parser.parse_args()

models = [
    'nerfacto'
]

if args.initialize:
    if args.videos_dir:
        cria_pastas(args.videos_dir)
else:
    if args.images_dir:
        main(args.images_dir, models, is_images=True)
    elif args.videos_dir:
        main(args.videos_dir, models)
# main("/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_4", models)
