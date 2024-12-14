import os

from auditor_utils import nerfstudio_export, read_info, write_info
import argparse

export_model = "poisson"
parser = argparse.ArgumentParser(description="Script with argparse options")
parser.add_argument("-vd", "--videos_dir", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)

# Parse arguments
args = parser.parse_args()

datasets = [os.path.join(args.videos_dir, path) for path in os.listdir(args.videos_dir)]

for dataset in datasets:
    info_path = (os.path.join(dataset, "info.json"))
    metrics_path = (os.path.join(dataset, "output_metrics_features.json"))

    tempo_export, gpu_export_vram, gpu_export_perc, ram_export = nerfstudio_export(export_model, os.path.join(dataset, 'output_nerfacto/'),  info_path, force_export=True)

    metrics = read_info(metrics_path)
    if not metrics[export_model]:
        metrics[export_model] = {}

    metrics[export_model]["tempo_export"] = tempo_export
    metrics[export_model]["gpu_export_max_vram"] = max(gpu_export_vram)
    metrics[export_model]["gpu_export_max_perc"] = max(gpu_export_perc)
    metrics[export_model]["ram_export_max"] = max(ram_export)
    write_info(metrics_path, metrics)
