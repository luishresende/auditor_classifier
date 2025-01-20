import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import struct
import matplotlib.cm as cm
import random
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import json
import cv2
from time import time, sleep
import subprocess
import pandas as pd
import io
import csv
import math
import sqlite3

def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '█'
    spaces = ' ' * (bar_length - int(progress * bar_length))
    print(f'\rProgress: [{arrow * int(progress * bar_length)}{spaces}] {progress * 100:.2f}%', end='', flush=True)

def compute_laplacian(images_paths):
    laplacians = []
    for k, image_path in enumerate(images_paths):
        if image_path.endswith('.png'):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacians.append(var)
            if k % 50 == 0:
                print_progress_bar(k, len(images_paths))
    laplacians = np.array(laplacians)
    return laplacians

def select_windows(frames_number, total_number_frames):
    space = np.linspace(0, total_number_frames, frames_number+1, dtype=int)
    return space[0:frames_number], space[1:]

def apaga_frames_com_mais_blur(frames_path, frames_number, laplacians):
    begs, ends = select_windows(frames_number, len(laplacians))
    for beg, end in zip(begs, ends):
        idx = laplacians[beg:end].argmax() + beg
        for j in range(beg, end):
            if j != idx:
                os.system('rm ' + frames_path + '/frame{:05d}.png'.format(j+1))

def preprocess_images(frames_path):
    images_paths = os.listdir(frames_path)
    images_paths = sorted(images_paths)
    images_paths = [os.path.join(frames_path, image_path) for image_path in images_paths]
    return images_paths

def extrai_frames_ffmpeg(parent_path, video_folder, video_path, target_resolution=(1280, 720)):
    output_folder = os.path.join(parent_path, video_folder, 'images_orig')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ffmpeg_command = f'ffmpeg -i "{os.path.join(parent_path, video_folder, video_path)}" ' \
                     f'-vf "scale={target_resolution[0]}:{target_resolution[1]}" ' \
                     f'"{os.path.join(output_folder, "frame%05d.png")}"'

    if len(os.listdir(output_folder)) == 0:
        os.system(ffmpeg_command)

def extrai_frames(parent_path, video_folder, video_path, frames_number, info_path):
    info = read_info(info_path)

    if not info["extract"]:
        extrai_frames_ffmpeg(parent_path, video_folder, video_path)
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images_orig')))
        info["extract"] = True
    if not info["delete_blurred"]:
        apaga_frames_com_mais_blur(os.path.join(parent_path, video_folder, 'images_orig'), frames_number, laplacians)
        info["delete_blurred"] = True
    if not info["colmap"] and not info["laplacians"]:
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images_orig')))
        info["laplacians"] = True
        info["lap_val"] = laplacians.tolist()
    elif not info["laplacians"]:
        laplacians = compute_laplacian(preprocess_images(os.path.join(parent_path, video_folder, 'images')))
        info["laplacians"] = True
        info["lap_val"] = laplacians.tolist()
    else:
        laplacians = info["lap_val"]
    write_info(info_path, info)
    return laplacians

def delete_colmap_partial_data(colmap_output_path):
    os.system("rm -rf " + os.path.join(colmap_output_path, "images"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_2"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_4"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "images_8"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "transforms.json"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "sparse_pc.ply"))
    os.system("rm -rf " + os.path.join(colmap_output_path, "colmap", "database.db"))

def get_num_images(images_input_path):
    images_sum = 0
    for x in os.listdir(images_input_path):
        if not os.path.isdir(x):
            images_sum += 1
    return images_sum

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file, num_images):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        Qs = [None] * num_images
        Ts = [None] * num_images
        image_names = []
        image_ids = []
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            image_ids.append(image_id)
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            image_names.append(image_name)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            Qs[image_id - 1] = qvec
            Ts[image_id - 1] = tvec
    
    return Qs, Ts, num_reg_images, image_names, image_ids

def return_maximum_size_reconstruction(colmap_output_path, num_images):
    num_reg_images_max = 0
    folder_max = None
    for folder in os.listdir(os.path.join(colmap_output_path, "colmap", "sparse")):
        if folder.isdigit():
            Q, T, num_reg_images, _, _ = read_images_binary(os.path.join(colmap_output_path, "colmap", "sparse", folder, "images.bin"), num_images)
            if num_reg_images > num_reg_images_max:
                num_reg_images_max = num_reg_images
                Qs = Q
                Ts = T
                folder_max = int(folder)
    
    return Qs, Ts, num_reg_images_max, folder_max

def is_wrong(colmap_output_path, images_path):
    diri_colmap = os.path.join(colmap_output_path, 'colmap/sparse')
    if len(os.listdir(diri_colmap)) == 0:
        return True
    
    # Get number of images extracted of the video
    num_images = get_num_images(images_path)

    # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
    _, _, num_reg_images_max, _ = return_maximum_size_reconstruction(colmap_output_path, num_images)
    if num_reg_images_max == num_images:
        return False
    return True

def delete_dir(path, diri):
    if os.path.exists(os.path.join(path, diri)):
        os.system('rm -rf ' + os.path.join(path, diri))

def delete_colmap_dirs(colmap_output_path):
    delete_dir(colmap_output_path, 'colmap')
    delete_colmap_partial_data(colmap_output_path)

# Function to get GPU usage
def get_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        res = result.stdout.strip()
        gpu_usage = res[:res.find(',')]
        gpu_percentage = res[res.find(',')+1:]
        return int(gpu_usage), int(gpu_percentage)
    except subprocess.CalledProcessError as e:
        print(f"Error querying GPU usage: {e}")
        return None, None
    
# Function to get GPU usage
def get_ram_usage():
    try:
        result = subprocess.run(
            ["free"],
            capture_output=True,
            text=True,
            check=True
        )
        ram_usage = result.stdout.strip()
        ram_data = io.StringIO(ram_usage)
        ram_df = pd.read_csv(ram_data, sep='\s+')
        ram_df = ram_df.loc['Mem.:']
        ram_usage = (ram_df['total'] - ram_df['disponível']) / 1e6 # In Gb
        return ram_usage
    except subprocess.CalledProcessError as e:
        print(f"Error querying RAM usage: {e}")
        return None

def choose_best_camera_model_and_refine_intrinsics(colmap_output_path, frames_parent_path):
    num_images = get_num_images(os.path.join(frames_parent_path, "images_orig"))
    _, _, _, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    if camera_model != 0:
        path = os.path.join(colmap_output_path, "colmap", "sparse", f"{camera_model}")
        path_0 = os.path.join(colmap_output_path, "colmap", "sparse", "0")
        path_1 = os.path.join(colmap_output_path, "colmap", "sparse", "_1")
        os.system(f"mv {path_0} {path_1}")
        os.system(f"mv {path} {path_0}")
        os.system(f"mv {path_1} {path}")
        os.system(f"colmap bundle_adjuster --input_path {path_0} --output_path {path_0} --BundleAdjustment.refine_principal_point 1")
        os.system(f"ns-process-data images --data {os.path.join(colmap_output_path, 'images_orig')} --output-dir {colmap_output_path} --matching-method exhaustive --skip-colmap --skip-image-processing")
    return camera_model

def run_command(cmd):
    gpu_vram = []
    gpu_perc = []
    ram = []

    process = subprocess.Popen(cmd)
    # Monitor GPU usage while the command is running
    try:
        while process.poll() is None:  # Check if process is still running
            gpu_usage, gpu_percentage = get_gpu_usage()
            ram_usage = get_ram_usage()
            if gpu_usage:
                gpu_vram.append(gpu_usage)
                gpu_perc.append(gpu_percentage)
            if ram_usage:
                ram.append(ram_usage)
            sleep(1)  # Adjust the interval as needed
    finally:
        process.wait()  # Ensure the process completes
        gpu_usage, gpu_percentage = get_gpu_usage()
        ram_usage = get_ram_usage()
        if gpu_usage:
            gpu_vram.append(gpu_usage)
            gpu_perc.append(gpu_percentage)
        if ram_usage:
            ram.append(ram_usage)
    return gpu_vram, gpu_perc, ram

def preprocess_data(frames_parent_path, colmap_output_path, colmap_limit, info_path, downscale):
    info = read_info(info_path)
    if not info["colmap"]:
        number_iterations = 0
        is_wrong_flag = True
        start = time()
        while is_wrong_flag and number_iterations < colmap_limit:
            delete_colmap_dirs(colmap_output_path)
            cmd = [
                "ns-process-data", "images", 
                "--data", os.path.join(frames_parent_path, "images_orig"), 
                "--output-dir", colmap_output_path, 
                "--matching-method", "exhaustive",
                "--no-refine-intrinsics",
                "--num-downscales", f'{downscale}'
            ]
            gpu_vram, gpu_perc, ram = run_command(cmd)
            is_wrong_flag = is_wrong(colmap_output_path, os.path.join(frames_parent_path, "images_orig"))
            number_iterations += 1
        
        camera_model = choose_best_camera_model_and_refine_intrinsics(colmap_output_path, frames_parent_path)

        end = time()
        sleep(1.0)
        tempo = end - start

        os.system('rm -rf ' + os.path.join(frames_parent_path, 'images_orig'))
        info["colmap"] = True
        info["gpu_colmap_vram"] = gpu_vram
        info["gpu_colmap_perc"] = gpu_perc
        info["ram_colmap"] = ram
        info["tempo_colmap"] = tempo
        info["colmap_tries"] = number_iterations
        info["camera_model"] = camera_model
    else:
        gpu_vram = info["gpu_colmap_vram"]
        gpu_perc = info["gpu_colmap_perc"]
        ram = info["ram_colmap"]
        tempo = info["tempo_colmap"]
        number_iterations = info["colmap_tries"]
        camera_model = info["camera_model"]
    write_info(info_path, info)
    return tempo, gpu_vram, gpu_perc, ram, number_iterations, camera_model

def nerfstudio_model(colmap_output_path, splatfacto_output_path, info_path, model, propert=None):
    info = read_info(info_path)
    if not info[model]["trained"]:
        start = time()
        cmd = [
            "ns-train", model,
            "--data", colmap_output_path,
            "--max-num-iterations", "49999",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.predict-normals", "True",
            "--output-dir", splatfacto_output_path,
            "nerfstudio-data", "--downscale-factor", str(propert.downscale_factor),
            "--train-split-fraction", str(propert.split_fraction)
        ]
        gpu_vram, gpu_perc, ram = run_command(cmd)
        end = time()
        sleep(1.0)
        tempo = end - start
        info[model]["trained"] = True
        info[model][f"gpu_train_vram"] = gpu_vram
        info[model][f"gpu_train_perc"] = gpu_perc
        info[model][f"ram_train"] = ram
        info[model][f"tempo_train"] = tempo
    else:
        gpu_vram = info[model][f"gpu_train_vram"]
        gpu_perc = info[model][f"gpu_train_perc"]
        ram = info[model][f"ram_train"]
        tempo = info[model][f"tempo_train"]
    write_info(info_path, info)
    return tempo, gpu_vram, gpu_perc, ram

def search_config(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if 'config.yml' in files:
            config_path = os.path.join(root, 'config.yml')
            return config_path
    return None  # Retorna None se o arquivo não for encontrado

def nerfstudio_export(model, nerf_output_path, info_path):
    info = read_info(info_path)
    output_dir = os.path.join(str(nerf_output_path), str(model))
    os.makedirs(output_dir, exist_ok=True)

    if not info[model]["exported"]:
        start = time()
        cmd = [
            "ns-export", model,
            "--load-config", search_config(nerf_output_path),
            "--output-dir", output_dir,
            "--obb-center", "0", "0", "0",
            "--obb-rotation", "0", "0", "0",
            "--obb-scale", "2", "2", "2"
        ]
        gpu_vram, gpu_perc, ram = run_command(cmd)
        end = time()
        sleep(1.0)
        tempo = end - start
        info[model]["exported"] = True
        info[model][f"gpu_export_vram"] = gpu_vram
        info[model][f"gpu_export_perc"] = gpu_perc
        info[model][f"ram_export"] = ram
        info[model][f"tempo_export"] = tempo
    else:
        gpu_vram = info[model][f"gpu_export_vram"]
        gpu_perc = info[model][f"gpu_export_perc"]
        ram = info[model][f"ram_export"]
        tempo = info[model][f"tempo_export"]
    write_info(info_path, info)
    return tempo, gpu_vram, gpu_perc, ram

def nerfstudio_model_evaluations(model_output_path, video_folder, destino_path, model, info_path, project_path, propert=None):
    info = read_info(info_path)
    elems = [*range(10000, propert.max_num_iterations, 10000)]
    elems.append(propert.max_num_iterations-1)
    if not info[model]["evaluations"]:
        psnr = []
        ssim = []
        lpips = []
        fps = []
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, '*', '*', 'nerfstudio_models', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, '*', '*'))
            sleep(1)
        
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, '*', '*', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, '*', '*', 'nerfstudio_models'))
            sleep(1)
            os.system('mkdir ' + destino_path)
            os.system('ns-eval --load-config ' + os.path.join(model_output_path, video_folder, '*', '*', 'config.yml') + ' --output-path ' + os.path.join(destino_path, f'eval_ckpt_{elem:09}.json'))
            with open(os.path.join(destino_path, f'eval_ckpt_{elem:09}.json')) as file:
                content = json.load(file)
                psnr.append(content['results']['psnr'])
                ssim.append(content['results']['ssim'])
                lpips.append(content['results']['lpips'])
                fps.append(content['results']['fps'])
        info[model]["evaluations"] = True
        info[model]["psnr"] = psnr
        info[model]["ssim"] = ssim
        info[model]["lpips"] = lpips
        info[model]["fps"] = fps
    else:
        psnr = info[model]["psnr"]
        ssim = info[model]["ssim"]
        lpips = info[model]["lpips"]
        fps = info[model]["fps"]
    elem_best = elems[lpips.index(min(lpips))]
    eval_path = os.path.join(destino_path, f'eval_ckpt_{elem_best:09}.json')
    get_eval_images(project_path, propert.split_fraction, eval_path, model)
    write_info(info_path, info)
    return psnr, ssim, lpips, fps

def get_eval_images(project_path, train_split_fraction, eval_path, ns_model, fontsize=15):
    db_path = os.path.join(project_path, 'colmap', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count descriptors
    cursor.execute("SELECT image_id, rows AS num_descriptors FROM keypoints;")
    descriptors = cursor.fetchall()
    
    conn.close()

    imagesd = []
    for image, row in descriptors:
        imagesd.append(image)
    imagesd = np.array(imagesd)

    idxs = models_images(os.path.join(project_path, 'colmap', 'sparse'), os.path.join(project_path, 'images'))
    num_images = len(idxs)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images
    i_eval = idxs[i_eval]

    metrics_list = get_metrics_list(eval_path)
    folder_report = f'report/{ns_model}'
    os.system(f'mkdir {os.path.join(project_path, folder_report)}')
    if metrics_list is not None:
        psnr = np.array([elem['psnr'] for elem in metrics_list])
        ssim = np.array([elem['ssim'] for elem in metrics_list])
        lpips = np.array([elem['lpips'] for elem in metrics_list])
        
        PSNR_limit = 25
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], psnr, 'b', linewidth=2)
        plt.plot(imagesd[i_eval], psnr, '.b', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [PSNR_limit, PSNR_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, PSNR_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, PSNR_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, PSNR_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, PSNR_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('PSNR', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation PSNR', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, 'PSNR_per_image.png'), bbox_inches='tight')

        SSIM_limit = 0.9
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], ssim, 'r', linewidth=2)
        plt.plot(imagesd[i_eval], ssim, '.r', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [SSIM_limit, SSIM_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, SSIM_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, SSIM_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, SSIM_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, SSIM_limit+0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('SSIM', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation SSIM', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, 'SSIM_per_image.png'), bbox_inches='tight')

        LPIPS_limit = 0.15
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(imagesd[i_eval], lpips, 'g', linewidth=2)
        plt.plot(imagesd[i_eval], lpips, '.g', markersize=8)
        plt.plot([imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05], [LPIPS_limit, LPIPS_limit], '--k', linewidth=3)
        arrow = FancyArrowPatch((imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, LPIPS_limit), (imagesd[i_eval[0]]-imagesd[i_eval[-1]]*0.05, LPIPS_limit-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, LPIPS_limit), (imagesd[i_eval[-1]]+imagesd[i_eval[-1]]*0.05, LPIPS_limit-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0])), arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
        plt.grid(True)
        plt.ylabel('LPIPS', fontsize=fontsize)
        plt.xlabel('Images', fontsize=fontsize)
        plt.title('Evaluation LPIPS', fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)  # Change xticks fontsize
        plt.tick_params(axis='y', labelsize=fontsize)  # Change yticks fontsize
        plt.savefig(os.path.join(project_path, folder_report, 'LPIPS_per_image.png'), bbox_inches='tight')
    else:
        with open(os.path.join(project_path, folder_report, 'ALERT.txt'), 'w') as filetext:
            filetext.write('Metrics per image not generated, because Nerfstudio code was not changed as shown in the file changes_nerfstudio_metric_per_image.txt')

def models_images(path_to_model_file, path_images):
    image_idss = []
    models = []

    for folder in sorted(os.listdir(path_to_model_file)):
        image_idsd = get_images_with_pose_ids(os.path.join(path_to_model_file, folder, 'images.bin'), path_images)
        image_idss.append(image_idsd)
        models.append(int(folder))
    idx = models.index(0)
    image_idsd = np.array(image_idss[idx])
    return np.where(image_idsd == 1)[0]

def get_metrics_list(eval_path):
    info = read_info(eval_path)
    if 'results_list' in info.keys():
        return info['results_list']
    else:
        return None

def compute_metrics(camera_positions, normals):
    # Percentage of normals looking to inside
    number_normals_to_inside = 0
    for camera_position, normal in zip(camera_positions, normals):
        cos_angle = (camera_position @ normal) / (np.linalg.norm(camera_position) * np.linalg.norm(normal))
        if cos_angle < 0:
            number_normals_to_inside += 1
    percentage_normals_to_inside = number_normals_to_inside / len(camera_positions)

    # Number of views
    thetas, phis = [], []
    for camera_position in camera_positions:
        _, theta, phi = cartesian_to_spherical(camera_position[0], camera_position[1], camera_position[2])
        thetas.append(theta)
        phis.append(phi)

    return percentage_normals_to_inside, thetas, phis

def plot_number_views(thetas, phis, M=10, N=20, centered=False, plot=True):
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Plot rectangles
    ij = np.zeros((2 * M, N))
    for theta, phi in zip(thetas, phis):
        i = np.floor(M * theta / np.pi)
        j = np.floor(N * phi / np.pi)
        rect = patches.Rectangle((i * np.pi / M, j * np.pi / N), np.pi / M, np.pi / N, color='g')
        if plot:
            ax.add_patch(rect)
        ij[int(i) + M][int(j)] = 1
    percentage_angle_views = sum(sum(ij)) / (2 * M * N)
    
    # Plot grids
    if plot:
        for j in range(N+1):
            ax.plot([-np.pi, np.pi], [j * np.pi/N, j * np.pi/N], 'k', linewidth=0.5)
        for i in range(-M, M+1):
            ax.plot([i * np.pi/M, i * np.pi/M], [0, np.pi], 'k', linewidth=0.5)

        # Plot points
        ax.scatter(thetas, phis, marker='.', c='r')

        if not centered:
            ax.set_title(f"{percentage_angle_views*100:.2f}% of view angles used when not centered")
        else:
            ax.set_title(f"{percentage_angle_views*100:.2f}% of view angles used when centered")

        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
    return percentage_angle_views

def cartesian_to_spherical(x, y, z):
    # Radial distance (r)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Azimuthal angle (theta)
    theta = np.arctan2(y, x)
    
    # Polar angle (phi)
    phi = np.arccos(z / r)
    
    return r, theta, phi

def return_camera_positions(Qs, Ts, centered=False):
    camera_positions = []
    normals = []
    for q, t in zip(Qs, Ts):
        if q is not None and t is not None:
            rot = R.from_quat(q)
            rot_matrix = rot.as_matrix()
            camera_positions.append(- rot_matrix.T @ t)
            v = rot_matrix.T @ np.array([0,0,1])
            normals.append(v / np.linalg.norm(v) / 2)
    
    # Center of video
    center = np.mean(np.array(camera_positions), axis=0)

    # Basis of the center of mass
    aux = random.sample(normals, len(normals))
    aux1 = np.mean(aux[:len(aux)//2], axis=0)
    aux1 /= np.linalg.norm(aux1)
    aux2 = np.mean(aux[len(aux)//2:], axis=0)
    aux2 /= np.linalg.norm(aux2)
    w = np.cross(aux1, aux2) / (np.linalg.norm(np.cross(aux1, aux2)))
    Rot = np.array([aux1, np.cross(w, aux1), w]).T
    Rotinv = np.linalg.inv(Rot)
    camera_positions_center = [x - center for x in camera_positions]
    camera_positions_center = [Rotinv @ x for x in camera_positions_center]
    normals_center = [Rotinv @ x for x in normals]
    if centered:
        return camera_positions_center, normals_center, [0,0,1], [0,0,0]
    else:
        return camera_positions, normals, w, center

def preprocess_evaluation_main(colmap_output_path, images_path):
    # Get number of images extracted of the video
    try:
        num_images = get_num_images(images_path[0])
    except FileNotFoundError:
        try:
            num_images = get_num_images(images_path[1])
        except FileNotFoundError:
            try:
                num_images = get_num_images(images_path[2])
            except FileNotFoundError:
                num_images = get_num_images(images_path[3])

    # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
    Qs, Ts, num_reg_images_max, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    
    # Get the camera positions and orientations
    camera_positions, normals, _, _ = return_camera_positions(Qs, Ts)
    camera_positions_center, normals_center, _, _ = return_camera_positions(Qs, Ts, True)

    # Compute metrics for the trajectory
    normals_inside, thetas, phis = compute_metrics(camera_positions, normals)
    normals_inside_center, thetas_center, phis_center = compute_metrics(camera_positions_center, normals_center)

    # Plot number of views
    percentage_angle_views = plot_number_views(thetas, phis, centered=False, plot=False)
    percentage_angle_views_center = plot_number_views(thetas_center, phis_center, centered=True, plot=False)

    return normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, num_reg_images_max / num_images, camera_model

def plot_matches_metrics(project_path):
    folder_report = 'report'
    os.system(f'mkdir {project_path}/{folder_report}')
    folder_report = 'report/colmap'
    os.system(f'mkdir {project_path}/{folder_report}')
    path_images = os.path.join(project_path, 'images')
    path_to_model_file = os.path.join(project_path, 'colmap', 'sparse')

    # COLMAP DATABASE
    db_path = os.path.join(project_path, 'colmap', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_id, rows AS num_descriptors FROM keypoints;")
    descriptors = cursor.fetchall()
    
    cursor.execute("SELECT pair_id, rows AS num_matches FROM matches;")
    total_matches = cursor.fetchall()

    cursor.execute("SELECT pair_id, config AS num_geometries FROM two_view_geometries;")
    geometries = cursor.fetchall()

    cursor.execute("SELECT image_id, name AS names FROM images;")
    image_names = cursor.fetchall()
    image_namesd = {}
    for id, name in image_names:
        image_namesd[id] = name
    
    conn.close()

    imagesd = []
    num_descriptors = []
    for image, row in descriptors:
        imagesd.append(image)
        num_descriptors.append(row)

    matchesd = np.zeros((len(imagesd),len(imagesd)))
    for pair, row in total_matches:
        im1 = pair // 2147483647
        im2 = pair % 2147483647
        matchesd[im1-1,im2-1] = row
        matchesd[im2-1,im1-1] = row
    
    matches_configd = np.zeros((len(imagesd),len(imagesd)))
    for pair, row in geometries:
        im1 = pair // 2147483647
        im2 = pair % 2147483647
        matches_configd[im1-1,im2-1] = row
        matches_configd[im2-1,im1-1] = row

    # Matches with cameras found
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'w']
    k = 0
    plt.figure()
    image_idss = []
    models = []
    nones = np.zeros_like(imagesd, dtype=int)
    for folder in sorted(os.listdir(path_to_model_file)):
        image_idsd = get_images_with_pose_ids(os.path.join(path_to_model_file, folder, 'images.bin'), path_images)
        plt.plot(
            imagesd, 
            100-np.count_nonzero(matches_configd==0, axis=0)*100/len(imagesd) * image_idsd, 
            '-', 
            color=colors[k], 
            label= 'model ' + folder, 
            linewidth=2
        )
        k += 1
        aux = image_idsd
        aux[np.isnan(aux)] = 0
        nones = nones | aux.astype(int)
        image_idss.append(image_idsd)
        models.append('model ' + folder)
    nones = nones.astype(float)
    nones[nones==1] = np.nan
    nones[nones==0] = 1
    if True in (nones==1):
        plt.plot(
            imagesd, 
            100-np.count_nonzero(matches_configd==0, axis=0)*100/len(imagesd) * nones, 
            'x', 
            color='k', 
            label= 'no model', 
            linewidth=2
        )
    plt.xlabel('Images')
    plt.ylabel('Percentage')
    plt.title('Percentage of images matches related with some camera type for the models found')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(project_path, folder_report, 'percentage_cameras_matches.png'), bbox_inches='tight')

    # Matches between images
    plt.figure(figsize=(8,8))
    plt.imshow(matchesd**0.25, cmap='hot')  # 'viridis' is the colormap; you can try 'gray', 'hot', etc.
    plt.colorbar()  # Add a colorbar to indicate value ranges
    plt.title(f"Matches between images")
    plt.xlabel('Images')
    plt.ylabel('Images')
    plt.savefig(os.path.join(project_path, folder_report, 'images_matches.png'), bbox_inches='tight')

    # List of images per model
    lists = []
    for model, image_idsd in zip(models, image_idss):
        aux = np.array(imagesd)[image_idsd == 1].astype(int).tolist()
        aux = [image_namesd[key] for key in aux]
        lists.append(aux)
    aux = np.array(imagesd)[nones == 1].astype(int).tolist()
    aux = [image_namesd[key] for key in aux]
    lists.append(aux)
    
    df = pd.DataFrame(lists).T.fillna(' ')
    models.append('no model')
    df.columns = models
    df.to_csv(os.path.join(project_path, folder_report, 'models_with_images.csv'), index=False, sep='\t')

def get_images_with_pose_ids(path_to_model_file, path_images):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    num_images = get_num_images(path_images)
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        Qs = [None] * num_images
        Ts = [None] * num_images
        image_names = []
        image_ids = np.ones((num_images,)) * np.nan
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            image_ids[image_id-1] = 1
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            image_names.append(image_name)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
    return image_ids

def init(parent_path, video_folder, is_images=False):
    if not os.path.exists(os.path.join(parent_path, video_folder, "info.json")):
        info = {
            "extract": is_images,
            "delete_blurred": is_images,
            "laplacians": False,
            "pilot": False,
            "colmap": False,
            "nerfacto": {
                "trained": False,
                "evaluations": False
            },
            "poisson": {
                "exported": False
            }
        }
        write_info(os.path.join(parent_path, video_folder, "info.json"), info)
    return os.path.join(parent_path, video_folder, "info.json")

def read_info(info_path):
    with open(info_path) as file:
        info = json.load(file)
    return info
    
def write_info(info_path, info):
    json_object = json.dumps(info, indent = 2)
    with open(info_path, 'w') as file:
        file.write(json_object)
        file.close()

def pipeline(parent_path, video_folder, video_path, pilot_output_path, colmap_output_path, splatfacto_output_path, models, is_images=False, export_model="poisson", propert=None):
    colmap_limit = 1
    elems = [*range(10000, propert.max_num_iterations, 10000)]
    elems.append(propert.max_num_iterations-1)

    frames_parent_path = os.path.join(parent_path, video_folder)
    images_path = os.path.join(frames_parent_path, 'images_orig')

    # Init
    info_path = init(parent_path, video_folder, is_images)

    # Extract frames and get laplacians
    laplacians = extrai_frames(parent_path, video_folder, video_path, propert.frames_number, info_path)


    images_path_8 = [
        os.path.join(frames_parent_path, f'images')
    ]

    # Pilot study with repetitions
    # pilot_study(repetition_number, frames_parent_path, pilot_output_path, info_path)


    # Preprocess dataset
    tempo_colmap, gpu_colmap_vram, gpu_colmap_perc, ram_colmap, number_iterations_colmap, camera_model = preprocess_data(frames_parent_path, colmap_output_path, colmap_limit, info_path, downscale=0)

    # Colmap evaluations
    normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, percentage_poses_found, _ = preprocess_evaluation_main(colmap_output_path, images_path_8)

    # Colmap pilot study evaluations
    # normals_inside_pilot, normals_inside_center_pilot, percentage_angle_views_pilot, percentage_angle_views_center_pilot, percentage_poses_found_pilot, camera_models_pilot = colmap_evaluation_pilot(os.path.join(frames_parent_path, pilot_output_path), images_path_8)

    output = {
            "lap_mean": np.mean(laplacians),
            "lap_max": max(laplacians),
            "lap_min": min(laplacians),
            "lap_median": np.median(laplacians),

            "tempo_colmap": tempo_colmap,
            "gpu_colmap_max_vram": max(gpu_colmap_vram),
            "gpu_colmap_max_perc": max(gpu_colmap_perc),
            "ram_colmap_max": max(ram_colmap),
            "number_iterations_colmap": number_iterations_colmap,

            "percentage_normals_inside": normals_inside,
            "percentage_normals_inside_center": normals_inside_center,
            "percentage_angle_views": percentage_angle_views,
            "percentage_angle_views_center": percentage_angle_views_center,
            "percentage_poses_found": percentage_poses_found,
            "camera_model": camera_model,

            # "percentage_normals_inside_pilot": normals_inside_pilot,
            # "percentage_normals_inside_center_pilot": normals_inside_center_pilot,
            # "percentage_angle_views_pilot": percentage_angle_views_pilot,
            # "percentage_angle_views_center_pilot": percentage_angle_views_center_pilot,
            # "percentage_poses_found_pilot": percentage_poses_found_pilot,
            # "camera_models_pilot": camera_models_pilot,
        }

    # Models
    for model in models:
        tempo_train, gpu_train_vram, gpu_train_perc, ram_train = nerfstudio_model(colmap_output_path, splatfacto_output_path + f"_{model}", info_path, model, propert=propert)
        tempo_export, gpu_export_vram, gpu_export_perc, ram_export = nerfstudio_export(export_model, splatfacto_output_path + f"_{model}", info_path)

        # Model evaluations
        psnr, ssim, lpips, fps = nerfstudio_model_evaluations(splatfacto_output_path + f"_{model}", video_folder, os.path.join(frames_parent_path, f'output_{model}', 'evaluations'), model, info_path, colmap_output_path, propert=propert)

        output[model] = {}

        output[model]["tempo_train"] = tempo_train
        output[model]["gpu_train_max_vram"] = max(gpu_train_vram)
        output[model]["gpu_train_max_perc"] = max(gpu_train_perc)
        output[model]["ram_train_max"] = max(ram_train)

        output[export_model] = {}
        output[export_model]["tempo_export"] = tempo_export
        output[export_model]["gpu_export_max_vram"] = max(gpu_export_vram)
        output[export_model]["gpu_export_max_perc"] = max(gpu_export_perc)
        output[export_model]["ram_export_max"] = max(ram_export)

        output[model]["psnr_train_max"] = max(psnr)
        output[model]["ssim_train_max"] = max(ssim)
        output[model]["lpips_train_min"] = min(lpips)
        output[model]["fps_train_min"] = min(fps)

        output[model]["psnr_train_max_ckpt"] = elems[psnr.index(max(psnr))]
        output[model]["ssim_train_max_ckpt"] = elems[ssim.index(max(ssim))]
        output[model]["lpips_train_min_ckpt"] = elems[lpips.index(min(lpips))]
        output[model]["fps_train_min_ckpt"] = elems[fps.index(min(fps))]
    
    return output
