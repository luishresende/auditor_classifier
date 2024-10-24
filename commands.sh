mkdir $1/colmap
DATASET_PATH=$1/colmap
cp -r $1/images_orig $DATASET_PATH
mv $DATASET_PATH/images_orig $DATASET_PATH/images

colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --SiftExtraction.use_gpu 1 --ImageReader.camera_model OPENCV

colmap sequential_matcher --database_path $DATASET_PATH/database.db --SiftMatching.use_gpu 1

mkdir $DATASET_PATH/sparse
colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse --Mapper.ba_global_function_tolerance 1e-6

mkdir $DATASET_PATH/dense
