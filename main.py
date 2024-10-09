from auditor_utils import *

def main():
    path = "/home/tafnes/Downloads/videos"
    for file in os.listdir(path):
        output = pipeline(
            path,
            file,
            file + ".MOV",
            "pilot",
            os.path.join(path, file),
            os.path.join(path, file, "output")
        )
        write_info(os.path.join(path, file, "output_metrics_features.json"), output)

main()