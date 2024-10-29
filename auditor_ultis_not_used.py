# def nerfstudio_colmap_w(frames_parent_path, colmap_output_path, colmap_limit, info_path):
#     info = read_info(info_path)
#     if not info["colmap"]:
#         number_iterations = 0
#         is_wrong_flag = True
#         gpu_vram = []
#         gpu_perc = []
#         ram = []
#         start = time()
#         while is_wrong_flag and number_iterations < colmap_limit:
#             delete_colmap_dirs(colmap_output_path)
#             cmd = [
#                 "bash", "commands.sh", frames_parent_path
#             ]
#             process = subprocess.Popen(cmd)
#             # Monitor GPU usage while the command is running
#             try:
#                 while process.poll() is None:  # Check if process is still running
#                     gpu_usage, gpu_percentage = get_gpu_usage()
#                     ram_usage = get_ram_usage()
#                     if gpu_usage:
#                         gpu_vram.append(gpu_usage)
#                         gpu_perc.append(gpu_percentage)
#                     if ram_usage:
#                         ram.append(ram_usage)
#                     sleep(1)  # Adjust the interval as needed
#             finally:
#                 process.wait()  # Ensure the process completes
#                 gpu_usage, gpu_percentage = get_gpu_usage()
#                 ram_usage = get_ram_usage()
#                 if gpu_usage:
#                     gpu_vram.append(gpu_usage)
#                     gpu_perc.append(gpu_percentage)
#                 if ram_usage:
#                     ram.append(ram_usage)
#             is_wrong_flag = is_wrong(colmap_output_path, os.path.join(frames_parent_path, "images_orig"))
#             number_iterations += 1

#         camera_model = escolhe_maior_modelo_de_camera(colmap_output_path, frames_parent_path)

#         num_images = get_num_images(os.path.join(frames_parent_path, "images_orig"))
#         os.system('rm -rf ' + os.path.join(frames_parent_path, 'images_orig'))

#         os.system(f'colmap bundle_adjuster --input_path {os.path.join(frames_parent_path, "colmap/sparse/0")} --output_path {os.path.join(frames_parent_path, "colmap/sparse/0")}')
#         os.system(f'colmap image_undistorter --image_path {os.path.join(frames_parent_path, "colmap/images")} --input_path {os.path.join(frames_parent_path, "colmap/sparse/0")} --output_path {os.path.join(frames_parent_path, "colmap/dense")} --output_type COLMAP --max_image_size 2000')

#         create_tsv_file(os.path.join(frames_parent_path, "colmap"), frames_parent_path[frames_parent_path.rfind('/')+1:], num_images)

#         end = time()
#         sleep(1.0)
#         tempo = end - start

#         info["colmap"] = True
#         info["gpu_colmap"] = gpu
#         info["ram_colmap"] = ram
#         info["tempo_colmap"] = tempo
#         info["colmap_tries"] = number_iterations
#         info["camera_model"] = camera_model
#     else:
#         gpu = info["gpu_colmap"]
#         ram = info["ram_colmap"]
#         tempo = info["tempo_colmap"]
#         number_iterations = info["colmap_tries"]
#         camera_model = 0
#     write_info(info_path, info)
#     return tempo, gpu, ram, number_iterations, camera_model

# def create_tsv_file(path, dataset, num_images):
#     _, _, _, image_names, image_ids = read_images_binary(os.path.join(path,'dense','sparse','images.bin'), num_images)
#     with open(os.path.join(path, 'brandenburg') + '.tsv', 'w') as file:
#         tsv_writer = csv.writer(file, delimiter='\t')

#         tsv_writer.writerow(['filename', 'id', 'split', 'dataset'])
#         k = 0
#         for image, id in zip(image_names, image_ids):
#             if k % 10 == 0:
#                 tsv_writer.writerow([image, id, 'test', dataset])
#             else:
#                 tsv_writer.writerow([image, id, 'train', dataset])
#             k += 1
#         file.close()

# def nerfstudio_splatfacto_w(colmap_output_path, splatfacto_output_path, info_path, model):
#     info = read_info(info_path)
#     if not info["splatfacto"]:
#         gpu = []
#         ram = []
#         start = time()
#         if model == 'splatfacto-w':
#             cmd = [
#                 "ns-train", 'splatfacto-w-light', 
#                 "--data", os.path.join(colmap_output_path, "colmap"), 
#                 "--max-num-iterations", "100000", 
#                 "--viewer.quit-on-train-completion", "True",
#                 "--steps-per-save", "10000", 
#                 "--save-only-latest-checkpoint", "False",
#                 "--output-dir", splatfacto_output_path,
#                 "colmap", "--data", os.path.join(colmap_output_path, "colmap"),
#                 # "nerf-w-data-parser-config", "--data", os.path.join(colmap_output_path, "colmap"),
#                 "--data-name", "brandenburg"
#             ]
#         elif model == 'splatfacto-w-big':
#             cmd = [
#                 "ns-train", 'splatfacto-w-light', 
#                 "--data", os.path.join(colmap_output_path, "colmap"), 
#                 "--max-num-iterations", "100000", 
#                 "--viewer.quit-on-train-completion", "True",
#                 "--steps-per-save", "10000", 
#                 "--save-only-latest-checkpoint", "False",
#                 "--output-dir", splatfacto_output_path,
#                 "--pipeline.model.cull_alpha_thresh", "0.005",
#                 "--pipeline.model.continue_cull_post_densification", "False",
#                 "--pipeline.model.densify-grad-thresh", "0.0006",
#                 "colmap", "--data", os.path.join(colmap_output_path, "colmap"),
#                 # "nerf-w-data-parser-config", "--data", os.path.join(colmap_output_path, "colmap"),
#                 "--data-name", "brandenburg"
#             ]
#         process = subprocess.Popen(cmd)
#         # Monitor GPU usage while the command is running
#         try:
#             while process.poll() is None:  # Check if process is still running
#                 gpu_usage = get_gpu_usage()
#                 ram_usage = get_ram_usage()
#                 if gpu_usage:
#                     gpu.append(gpu_usage)
#                 if ram_usage:
#                     ram.append(ram_usage)
#                 sleep(1)  # Adjust the interval as needed
#         finally:
#             process.wait()  # Ensure the process completes
#             gpu_usage = get_gpu_usage()
#             ram_usage = get_ram_usage()
#             if gpu_usage:
#                 gpu.append(gpu_usage)
#             if ram_usage:
#                 ram.append(ram_usage)
#         end = time()
#         sleep(1.0)
#         tempo = end - start
#         info["splatfacto"] = True
#         info["gpu_train"] = gpu
#         info["ram_train"] = ram
#         info["tempo_train"] = tempo
#     else:
#         gpu = info["gpu_train"]
#         ram = info["ram_train"]
#         tempo = info["tempo_train"]
#     write_info(info_path, info)
#     return tempo, gpu, ram

# def nerfstudio_evaluations_w(model_output_path, destino_path, model, info_path):
#     info = read_info(info_path)
#     if not info["evaluations"]:
#         elems = [*range(5000, 55000, 5000)]
#         elems.append(54999)
#         psnr = []
#         ssim = []
#         lpips = []
#         for elem in elems:
#             os.system('mv ' + os.path.join(model_output_path, 'colmap', model, '*', 'nerfstudio_models', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, 'colmap', model, '*'))
#             sleep(1)
        
#         for elem in elems:
#             os.system('mv ' + os.path.join(model_output_path, 'colmap', model, '*', f'step-{elem:09}.ckpt') + ' ' + os.path.join(model_output_path, 'colmap', model, '*', 'nerfstudio_models'))
#             sleep(1)
#             os.system('mkdir ' + destino_path)
#             os.system('ns-eval --load-config ' + os.path.join(model_output_path, 'colmap', model, '*', 'config.yml') + ' --output-path ' + os.path.join(destino_path, f'eval_ckpt{elem}.json'))
#             with open(os.path.join(destino_path, f'eval_ckpt{elem}.json')) as file:
#                 content = json.load(file)
#                 psnr.append(content['results']['psnr'])
#                 ssim.append(content['results']['ssim'])
#                 lpips.append(content['results']['lpips'])
#         info["evaluations"] = True
#         info["psnr"] = psnr
#         info["ssim"] = ssim
#         info["lpips"] = lpips
#     else:
#         psnr = info["psnr"]
#         ssim = info["ssim"]
#         lpips = info["lpips"]
#     write_info(info_path, info)
#     return psnr, ssim, lpips

# def colmap_evaluation_main_w(colmap_output_path, images_path):
#     # Get number of images extracted of the video
#     try:
#         num_images = get_num_images(images_path[0])
#     except FileNotFoundError:
#         try:
#             num_images = get_num_images(images_path[1])
#         except FileNotFoundError:
#             try:
#                 num_images = get_num_images(images_path[2])
#             except FileNotFoundError:
#                 num_images = get_num_images(images_path[3])

#     # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
#     Qs, Ts, num_reg_images_max, camera_model = return_maximum_size_reconstruction(colmap_output_path, num_images)
    
#     # Get the camera positions and orientations
#     camera_positions, normals, _, _ = return_camera_positions(Qs, Ts)
#     camera_positions_center, normals_center, _, _ = return_camera_positions(Qs, Ts, True)

#     # Compute metrics for the trajectory
#     normals_inside, thetas, phis = compute_metrics(camera_positions, normals)
#     normals_inside_center, thetas_center, phis_center = compute_metrics(camera_positions_center, normals_center)

#     # Plot number of views
#     percentage_angle_views = plot_number_views(thetas, phis, centered=False, plot=False)
#     percentage_angle_views_center = plot_number_views(thetas_center, phis_center, centered=True, plot=False)

#     return normals_inside, normals_inside_center, percentage_angle_views, percentage_angle_views_center, num_reg_images_max / num_images, camera_model

# def colmap_evaluation_pilot(pilot_path, images_path):
#     normals_inside_vec = []
#     normals_inside_center_vec = []
#     percentage_angle_views_vec = []
#     percentage_angle_views_center_vec = []
#     percentage_poses_found_vec = []
#     camera_model_vec = []

#     # Get number of images extracted of the video
#     try:
#         num_images = get_num_images(images_path[0])
#     except FileNotFoundError:
#         try:
#             num_images = get_num_images(images_path[1])
#         except FileNotFoundError:
#             try:
#                 num_images = get_num_images(images_path[2])
#             except FileNotFoundError:
#                 num_images = get_num_images(images_path[3])
#     for folder in os.listdir(pilot_path):
#         # Get the quaternions and translation arrays from the sparse model with the most quantity of poses found
#         Qs, Ts, num_reg_images_max, camera_model = return_maximum_size_reconstruction(os.path.join(pilot_path, folder), num_images)
        
#         # Get the camera positions and orientations
#         camera_positions, normals, _, _ = return_camera_positions(Qs, Ts)
#         camera_positions_center, normals_center, _, _ = return_camera_positions(Qs, Ts, True)

#         # Compute metrics for the trajectory
#         normals_inside, thetas, phis = compute_metrics(camera_positions, normals)
#         normals_inside_center, thetas_center, phis_center = compute_metrics(camera_positions_center, normals_center)

#         # Plot number of views
#         percentage_angle_views = plot_number_views(thetas, phis, centered=False, plot=False)
#         percentage_angle_views_center = plot_number_views(thetas_center, phis_center, centered=True, plot=False)

#         normals_inside_vec.append(normals_inside)
#         normals_inside_center_vec.append(normals_inside_center)
#         percentage_angle_views_vec.append(percentage_angle_views)
#         percentage_angle_views_center_vec.append(percentage_angle_views_center)
#         percentage_poses_found_vec.append(num_reg_images_max / num_images)
#         camera_model_vec.append(camera_model)

#     return normals_inside_vec, normals_inside_center_vec, percentage_angle_views_vec, percentage_angle_views_center_vec, percentage_poses_found_vec, camera_model_vec

# def pilot_study(repetition_number, frames_parent_path, pilot_output_path, info_path):
#     info = read_info(info_path)
#     if not info["pilot"]:
#         os.system("mkdir " + os.path.join(frames_parent_path, pilot_output_path))
#         for k in range(repetition_number):
#             os.system("mkdir " + os.path.join(frames_parent_path, pilot_output_path, f"{k}"))
#             sleep(0.5)
#             os.system("ns-process-data images --data " + os.path.join(frames_parent_path, "images_orig") + " --output-dir " + os.path.join(frames_parent_path, pilot_output_path, f"{k}") +  " --matching-method sequential")
#             sleep(0.5)
#             delete_colmap_partial_data(os.path.join(frames_parent_path, pilot_output_path, f"{k}"))
#         info["pilot"] = True
#     write_info(info_path, info)
