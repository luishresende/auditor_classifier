import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import struct
import matplotlib.cm as cm
import random
import matplotlib.patches as patches
import json
import cv2
from time import time, sleep
import subprocess
import pandas as pd
import io

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

def apaga_frames_com_mais_blur(frames_path, frames_number, laplacians):
    divide = len(laplacians) // frames_number
    for i in range(len(laplacians) // divide):
        idx = laplacians[i * divide:(i + 1) * divide].argmax() + i * divide
        for j in range(i * divide, (i + 1) * divide):
            if j != idx:
                os.system('rm ' + frames_path + '/frame{:05d}.png'.format(j+1))
        print_progress_bar(i, len(laplacians) // divide)
    if divide * (len(laplacians) // divide) < len(laplacians):
        idx = laplacians[(len(laplacians) // divide) * divide:].argmax() + (i+1) * divide
        for j in range((len(laplacians) // divide) * divide,len(laplacians)):
            if j != idx:
                os.system('rm ' + frames_path + '/frame{:05d}.png'.format(j+1))

def preprocess_images(frames_path):
    images_paths = os.listdir(frames_path)
    images_paths = sorted(images_paths)
    images_paths = [os.path.join(frames_path, image_path) for image_path in images_paths]
    return images_paths

def extrai_frames_ffmpeg(parent_path, video_folder, video_path):
    if not os.path.exists(os.path.join(parent_path, video_folder, 'images_orig')):
        os.system('mkdir ' + os.path.join(parent_path, video_folder, 'images_orig'))
        os.system('ffmpeg -i ' + os.path.join(parent_path, video_folder, video_path) + ' ' + os.path.join(parent_path, video_folder, 'images_orig') + r'/frame%5d.png')
    else:
        if len(os.listdir(os.path.join(parent_path, video_folder, 'images_orig'))) == 0:
            os.system('ffmpeg -i ' + os.path.join(parent_path, video_folder, video_path) + ' ' + os.path.join(parent_path, video_folder, 'images_orig') + r'/frame%5d.png')

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

def pilot_study(repetition_number, frames_parent_path, pilot_output_path, info_path):
    info = read_info(info_path)
    if not info["pilot"]:
        os.system("mkdir " + os.path.join(frames_parent_path, pilot_output_path))
        for k in range(repetition_number):
            os.system("mkdir " + os.path.join(frames_parent_path, pilot_output_path, f"{k}"))
            sleep(0.5)
            os.system("ns-process-data images --data " + os.path.join(frames_parent_path, "images_orig") + " --output-dir " + os.path.join(frames_parent_path, pilot_output_path, f"{k}") +  " --matching-method sequential")
            sleep(0.5)
            delete_colmap_partial_data(os.path.join(frames_parent_path, pilot_output_path, f"{k}"))
        info["pilot"] = True
    write_info(info_path, info)

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
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            Qs[image_id - 1] = qvec
            Ts[image_id - 1] = tvec
    
    return Qs, Ts, num_reg_images

def return_maximum_size_reconstruction(colmap_output_path, num_images):
    num_reg_images_max = 0
    folder_max = None
    for folder in os.listdir(os.path.join(colmap_output_path, "colmap", "sparse")):
        if folder.isdigit():
            Q, T, num_reg_images = read_images_binary(os.path.join(colmap_output_path, "colmap", "sparse", folder, "images.bin"), num_images)
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
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_usage = result.stdout.strip()
        return int(gpu_usage)
    except subprocess.CalledProcessError as e:
        print(f"Error querying GPU usage: {e}")
        return None
    
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

def escolhe_maior_modelo_de_camera(colmap_output_path, frames_parent_path):
    num_images = get_num_images(os.path.join(frames_parent_path, "images_orig"))
    _, _, _, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    if camera_model != 0:
        path = os.path.join(colmap_output_path, "colmap", "sparse", f"{camera_model}")
        path_0 = os.path.join(colmap_output_path, "colmap", "sparse", "0")
        path_1 = os.path.join(colmap_output_path, "colmap", "sparse", "_1")
        os.system(f"mv {path_0} {path_1}")
        os.system(f"mv {path} {path_0}")
        os.system(f"mv {path_1} {path}")
        os.system(f"ns-process-data images --data {os.path.join(colmap_output_path, "images_orig")} --output-dir {colmap_output_path} --matching-method exhaustive --skip-colmap --skip-image-processing")

def nerfstudio_colmap(frames_parent_path, colmap_output_path, colmap_limit, info_path):
    info = read_info(info_path)
    if not info["colmap"]:
        number_iterations = 0
        is_wrong_flag = True
        gpu = []
        ram = []
        start = time()
        while is_wrong_flag and number_iterations < colmap_limit:
            delete_colmap_dirs(colmap_output_path)
            cmd = [
                "ns-process-data", "images", 
                "--data", os.path.join(frames_parent_path, "images_orig"), 
                "--output-dir", colmap_output_path, 
                "--matching-method", "exhaustive"
            ]
            process = subprocess.Popen(cmd)
            # Monitor GPU usage while the command is running
            try:
                while process.poll() is None:  # Check if process is still running
                    gpu_usage = get_gpu_usage()
                    ram_usage = get_ram_usage()
                    if gpu_usage:
                        gpu.append(gpu_usage)
                    if ram_usage:
                        ram.append(ram_usage)
                    sleep(1)  # Adjust the interval as needed
            finally:
                process.wait()  # Ensure the process completes
                gpu_usage = get_gpu_usage()
                ram_usage = get_ram_usage()
                if gpu_usage:
                    gpu.append(gpu_usage)
                if ram_usage:
                    ram.append(ram_usage)
            is_wrong_flag = is_wrong(colmap_output_path, os.path.join(frames_parent_path, "images_orig"))
            number_iterations += 1
        end = time()
        sleep(1.0)
        tempo = end - start

        escolhe_maior_modelo_de_camera(colmap_output_path, frames_parent_path)

        os.system('rm -rf ' + os.path.join(frames_parent_path, 'images_orig'))
        info["colmap"] = True
        info["gpu_colmap"] = gpu
        info["ram_colmap"] = ram
        info["tempo_colmap"] = tempo
        info["colmap_tries"] = number_iterations
    else:
        gpu = info["gpu_colmap"]
        ram = info["ram_colmap"]
        tempo = info["tempo_colmap"]
        number_iterations = info["colmap_tries"]
    write_info(info_path, info)
    return tempo, gpu, ram, number_iterations

def nerfstudio_splatfacto(colmap_output_path, splatfacto_output_path, info_path):
    info = read_info(info_path)
    if not info["splatfacto"]:
        gpu = []
        ram = []
        start = time()
        cmd = [
            "ns-train", "splatfacto", 
            "--data", colmap_output_path, 
            "--max-num-iterations", "100000", 
            "--viewer.quit-on-train-completion", "True",
            "--steps-per-save", "10000", 
            "--save-only-latest-checkpoint", "False",
            "--output-dir", splatfacto_output_path
        ]
        process = subprocess.Popen(cmd)
        # Monitor GPU usage while the command is running
        try:
            while process.poll() is None:  # Check if process is still running
                gpu_usage = get_gpu_usage()
                ram_usage = get_ram_usage()
                if gpu_usage:
                    gpu.append(gpu_usage)
                if ram_usage:
                    ram.append(ram_usage)
                sleep(1)  # Adjust the interval as needed
        finally:
            process.wait()  # Ensure the process completes
            gpu_usage = get_gpu_usage()
            ram_usage = get_ram_usage()
            if gpu_usage:
                gpu.append(gpu_usage)
            if ram_usage:
                ram.append(ram_usage)
        end = time()
        sleep(1.0)
        tempo = end - start
        info["splatfacto"] = True
        info["gpu_train"] = gpu
        info["ram_train"] = ram
        info["tempo_train"] = tempo
    else:
        gpu = info["gpu_train"]
        ram = info["ram_train"]
        tempo = info["tempo_train"]
    write_info(info_path, info)
    return tempo, gpu, ram

def nerfstudio_evaluations(model_output_path, video_folder, destino_path, model, info_path):
    info = read_info(info_path)
    if not info["evaluations"]:
        elems = [*range(10000, 100000, 10000)]
        elems.append(99999)
        psnr = []
        ssim = []
        lpips = []
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, model, '*', 'nerfstudio_models', f'step-0000{elem}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, model, '*'))
            sleep(1)
        
        for elem in elems:
            os.system('mv ' + os.path.join(model_output_path, video_folder, model, '*', f'step-0000{elem}.ckpt') + ' ' + os.path.join(model_output_path, video_folder, model, '*', 'nerfstudio_models'))
            sleep(1)
            os.system('mkdir ' + destino_path)
            os.system('ns-eval --load-config ' + os.path.join(model_output_path, video_folder, model, '*', 'config.yml') + ' --output-path ' + os.path.join(destino_path, f'eval_ckpt{elem}.json'))
            with open(os.path.join(destino_path, f'eval_ckpt{elem}.json')) as file:
                content = json.load(file)
                psnr.append(content['results']['psnr'])
                ssim.append(content['results']['ssim'])
                lpips.append(content['results']['lpips'])
        info["evaluations"] = True
        info["psnr"] = psnr
        info["ssim"] = ssim
        info["lpips"] = lpips
    else:
        psnr = info["psnr"]
        ssim = info["ssim"]
        lpips = info["lpips"]
    write_info(info_path, info)
    return psnr, ssim, lpips

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

def colmap_evaluation_main(colmap_output_path, images_path):
    # Get number of images extracted of the video
    num_images = get_num_images(images_path)

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

def colmap_evaluation_pilot(pilot_path, images_path):
    normals_inside_vec = []
    normals_inside_center_vec = []
    percentage_angle_views_vec = []
    percentage_angle_views_center_vec = []
    percentage_poses_found_vec = []
    camera_model_vec = []

    # Get number of images extracted of the video
    num_images = get_num_images(images_path)
    for folder in os.listdir(pilot_path):
        # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
        Qs, Ts, num_reg_images_max, camera_model = return_maximum_size_reconstruction(os.path.join(pilot_path, folder), num_images)
        
        # Get the camera positions and orientations
        camera_positions, normals, _, _ = return_camera_positions(Qs, Ts)
        camera_positions_center, normals_center, _, _ = return_camera_positions(Qs, Ts, True)

        # Compute metrics for the trajectory
        normals_inside, thetas, phis = compute_metrics(camera_positions, normals)
        normals_inside_center, thetas_center, phis_center = compute_metrics(camera_positions_center, normals_center)

        # Plot number of views
        percentage_angle_views = plot_number_views(thetas, phis, centered=False, plot=False)
        percentage_angle_views_center = plot_number_views(thetas_center, phis_center, centered=True, plot=False)

        normals_inside_vec.append(normals_inside)
        normals_inside_center_vec.append(normals_inside_center)
        percentage_angle_views_vec.append(percentage_angle_views)
        percentage_angle_views_center_vec.append(percentage_angle_views_center)
        percentage_poses_found_vec.append(num_reg_images_max / num_images)
        camera_model_vec.append(camera_model)

    return normals_inside_vec, normals_inside_center_vec, percentage_angle_views_vec, percentage_angle_views_center_vec, percentage_poses_found_vec, camera_model_vec

def init(parent_path, video_folder):
    if not os.path.exists(os.path.join(parent_path, video_folder, "info.json")):
        info = {
            "extract": False,
            "delete_blurred": False,
            "laplacians": False,
            "pilot": False,
            "colmap": False,
            "splatfacto": False,
            "evaluations": False
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

def pipeline(parent_path, video_folder, video_path, pilot_output_path, colmap_output_path, splatfacto_output_path):
    repetition_number = 10
    frames_number = 300
    colmap_limit = 3
    frames_parent_path = os.path.join(parent_path, video_folder)
    images_path = os.path.join(frames_parent_path, 'images_orig')
    images_path_8 = os.path.join(frames_parent_path, 'images_8')

    info_path = init(parent_path, video_folder)
    laplacians = extrai_frames(parent_path, video_folder, video_path, frames_number, info_path)
    os.system("conda activate nerfstudio")
    sleep(1.0)
    pilot_study(repetition_number, frames_parent_path, pilot_output_path, info_path)
    tempo_colmap, gpu_colmap, ram_colmap, number_iterations_colmap = nerfstudio_colmap(frames_parent_path, colmap_output_path, colmap_limit, info_path)
    tempo_train, gpu_train, ram_train = nerfstudio_splatfacto(colmap_output_path, splatfacto_output_path, info_path)
    psnr, ssim, lpips = nerfstudio_evaluations(splatfacto_output_path, video_folder, os.path.join(frames_parent_path, 'evaluations'), 'splatfacto', info_path)
    os.system("conda deactivate")
    sleep(1.0)
    normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, percentage_poses_found, camera_model = colmap_evaluation_main(colmap_output_path, images_path_8)
    normals_inside_pilot, normals_inside_center_pilot, percentage_angle_views_pilot, percentage_angle_views_center_pilot, percentage_poses_found_pilot, camera_models_pilot = colmap_evaluation_pilot(os.path.join(frames_parent_path, pilot_output_path), images_path_8)

    output = {
        "lap_mean": np.mean(laplacians), 
        "lap_max": max(laplacians), 
        "lap_min": min(laplacians), 
        "lap_median": np.median(laplacians),
        
        "tempo_colmap": tempo_colmap, 
        "gpu_colmap_max": max(gpu_colmap), 
        "ram_colmap_max": max(ram_colmap), 
        "number_iterations_colmap": number_iterations_colmap,

        "tempo_train": tempo_train, 
        "gpu_train_max": max(gpu_train), 
        "ram_train_max": max(ram_train),

        "psnr_max": max(psnr), 
        "ssim_max": max(ssim), 
        "lpips_min": min(lpips),

        "percentage_normals_inside": normals_inside, 
        "percentage_normals_inside_center": normals_inside_center, 
        "percentage_angle_views": percentage_angle_views, 
        "percentage_angle_views_center": percentage_angle_views_center, 
        "percentage_poses_found": percentage_poses_found,
        "camera_model": camera_model,

        "percentage_normals_inside_pilot": normals_inside_pilot,
        "percentage_normals_inside_center_pilot": normals_inside_center_pilot, 
        "percentage_angle_views_pilot": percentage_angle_views_pilot, 
        "percentage_angle_views_center_pilot": percentage_angle_views_center_pilot, 
        "percentage_poses_found_pilot": percentage_poses_found_pilot,
        "camera_models_pilot": camera_models_pilot
    }
    return output