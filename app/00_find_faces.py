import os

from utils.face_utils import FaceDetector

models_root_dir = os.path.join(os.getenv("HOME"), "models")
photos_root_dir = os.path.join(os.getenv("HOME"), "photos")
face_det_model_path = os.path.join(
    models_root_dir, "mmod_human_face_detector.dat"
)
predictor_path = os.path.join(
    models_root_dir, "shape_predictor_68_face_landmarks.dat"
)
face_rec_model_path = os.path.join(
    models_root_dir, "dlib_face_recognition_resnet_model_v1.dat"
)

fd = FaceDetector(
    face_det_model_path=face_det_model_path,
    predictor_path=predictor_path,
    face_rec_model_path=face_rec_model_path,
    images_root_dir=photos_root_dir,
)

fd.collect_image_file_paths()
fd.load_images_batch()
fd.write_manifest()
fd.describe_from_manifest()
fd.cluster_faces()
fd.save_cluster_chips()
