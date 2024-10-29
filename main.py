from auditor_utils import *

def cria_pastas(path):
    for files in os.listdir(path):
        if not os.path.isdir(os.path.join(path, files)):
            os.system(f"mkdir {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")
            os.system(f"mv {os.path.join(path, files)} {os.path.join(path, files)[:os.path.join(path, files).rfind('.')]}")

def main(path, model):
    for file in os.listdir(path):
        try:
            output = pipeline(
                path,
                file,
                file + ".mp4",
                "pilot",
                os.path.join(path, file),
                os.path.join(path, file, "output"),
                model
            )
        except:
            output = pipeline(
                path,
                file,
                file + ".MOV",
                "pilot",
                os.path.join(path, file),
                os.path.join(path, file, "output"),
                model
            )
        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

path = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_1"
# cria_pastas(path)
main(path, 'splatfacto-w-light')

# path = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_1_W_big"
# # cria_pastas(path)
# main(path, 'splatfacto-w-big')