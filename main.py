from auditor_utils import *

def main():
    path = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Dataset_Lamia_1"
    for file in os.listdir(path):
        output = pipeline(
            path,
            file,
            file + ".mp4",
            "pilot",
            os.path.join(path, file),
            os.path.join(path, file, "output")
        )
        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

main()