from auditor_utils import *
import argparse

def cria_pastas(path):
    for files in os.listdir(path):
        if not os.path.isdir(os.path.join(path, files)):
            os.system(f"mkdir {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
            os.system(f"mv {os.path.join(path, files)} {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")

def main(path, models):
    for file in os.listdir(path):
        try:
            output = pipeline(
                path,
                file,
                file + ".mp4",
                "pilot",
                os.path.join(path, file),
                os.path.join(path, file, "output"),
                models
            )
        except:
            output = pipeline(
                path,
                file,
                file + ".MOV",
                "pilot",
                os.path.join(path, file),
                os.path.join(path, file, "output"),
                models
            )
        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

parser = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parser.add_argument("-v", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", required=True)
parser.add_argument("-i", "--initialize", type=bool, help="To initialize the videos folder.", default=False)

# Parse arguments
args = parser.parse_args()

models = [
    'nerfacto',
    'nerfacto-big',
    'splatfacto',
    'splatfacto-big',
    'splatfacto-w',
    'splatfacto-w-light'
]

if args.initialize:
    cria_pastas(args.videos_dir)
else:
    main(args.videos_dir, models)
# main("/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_4", models)
