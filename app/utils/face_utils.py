import os
import numpy as np
import dlib
from logging import basicConfig, info, debug
import glob
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import tensorflow as tf

import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

tf.config.experimental.set_visible_devices([], "GPU")


class FaceDetector:
    def __init__(
        self,
        face_det_model_path,
        predictor_path,
        face_rec_model_path,
        images_root_dir=os.getcwd(),
        debug_level="INFO",
    ):
        basicConfig(level=debug_level)
        self.dlib_gpus = dlib.cuda.get_num_devices()
        info("dlib BLAS Optimization: {}".format(dlib.DLIB_USE_BLAS))
        info("dlib LAPACK Optimization: {}".format(dlib.DLIB_USE_LAPACK))
        info("dlib CUDA Optimization: {}".format(dlib.DLIB_USE_CUDA))
        info("dlib CUDA Devices: {}".format(self.dlib_gpus))

        self.images_root_dir = images_root_dir
        self.images_file_list = []

        stamp = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.dest_file_name = os.path.join(
            self.images_root_dir, "face_db_" + stamp + ".json"
        )

        try:
            info(face_det_model_path)
            self.detector = dlib.cnn_face_detection_model_v1(
                face_det_model_path
            )
            self.mmod = True
            info("Using Convolutional Neural Network method.")
        except:
            self.detector = dlib.get_frontal_face_detector()
            self.mmod = False
            info("Using Histogram of Oriented Gradients method.")

        self.sp = dlib.shape_predictor(predictor_path)
        self.shapes = []
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.descriptors = []
        self.images = []

    def collect_image_file_paths(self):
        self.images_file_list = index_directory(self.images_root_dir)
        self.n_images = len(self.images_file_list)
        info("Found {:,} images".format(self.n_images))

    def load_image_file(self, file_path):

        try:
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img, channels=3)
            return img.numpy()
        except:
            return None

    def load_images_batch(self):
        list_manifest = run_apply_async_multiprocessing(
            self.detect_faces, self.images_file_list
        )
        self.manifest = {"manifest": list_manifest}

    def detect_faces(self, filename):
        img = self.load_image_file(filename)
        file_dict = {}
        file_dict["file_path"] = filename
        file_dict["face_metadata"] = []
        self.dets = self.detector(img, 1)
        file_dict["face_count"] = len(self.dets)
        for k, d in enumerate(self.dets):
            face_dict = {}
            idx = "face_{}".format(k)
            face_dict["id"] = idx
            if self.mmod:
                face_dict["confidence"] = d.confidence
                face_dict["dets_ltrb"] = [
                    d.rect.left(),
                    d.rect.top(),
                    d.rect.right(),
                    d.rect.bottom(),
                ]
            else:
                # HOG has no confidence measure.
                face_dict["confidence"] = 0.0
                face_dict["dets_ltrb"] = [
                    d.left(),
                    d.top(),
                    d.right(),
                    d.bottom(),
                ]
            file_dict["face_metadata"].append(face_dict)

        # with open(self.dest_file_name, "a") as outfile1:
        #     json.dump(file_dict, outfile1)
        #     outfile1.write("\n")

        return file_dict

    def write_manifest(self):
        with open(self.dest_file_name, "w") as outfile1:
            json.dump(self.manifest, outfile1)

    def read_manifest(self, json_file=None):
        if json_file:
            self.manifest = open(json_file)
        self.manifest = [json.loads(i) for i in self.manifest]

    def describe_from_manifest(self):
        run_apply_async_multiprocessing(self.describe_face, self.manifest)

    def describe_face(self, entry):
        image_file = entry["file_path"]
        img = self.load_image_file(image_file)
        for face in entry["face_metadata"]:
            l, t, r, b = face["dets_ltrb"]
            d = dlib.rectangle(l, t, r, b)
            shape = self.sp(img, d)
            facedesc = self.facerec.compute_face_descriptor(img, shape)
            self.shapes.append(shape)
            self.descriptors.append(facedesc)
            self.images.append((image_file, shape))

    def cluster_faces(self):
        self.labels = dlib.chinese_whispers_clustering(self.descriptors, 0.5)
        num_classes = len(set(self.labels))
        info("Number of clusters: {:,}".format(num_classes))
        self.indices = []
        for i, label in enumerate(self.labels):
            self.indices.append(i)

    def maybe_make_dir(self, output_folder_path):
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
            debug("Created directory {}".format(output_folder_path))

    def save_cluster_chips(self, root_dir=None, size=224, padding=0.25):
        if not root_dir:
            root_dir = self.images_root_dir

        cluster_root_dir = os.path.join(root_dir, "face_chip_clusters")
        self.maybe_make_dir(cluster_root_dir)

        for label_idx in list(set(self.labels)):
            label_dir = os.path.join(cluster_root_dir, str(label_idx))
            self.maybe_make_dir(label_dir)

        for i, index in enumerate(self.indices):
            image_file, shape = self.images[index]
            img = self.load_image_file(image_file)
            l = self.labels[index]
            output_folder_path = os.path.join(cluster_root_dir, str(l))
            file_path = os.path.join(output_folder_path, "face_" + str(i))
            dlib.save_face_chip(img, shape, file_path, size=224, padding=0.25)


def run_apply_async_multiprocessing(
    func, iterable, processes=int(cpu_count() * 32)
):
    results = []
    with ThreadPoolExecutor(max_workers=processes) as executor:
        future_to_iter = {executor.submit(func, i): i for i in iterable}
        for future in as_completed(future_to_iter):
            promise = future_to_iter[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                info("%r generated an exception: %s" % (promise, exc))

    return results


def index_directory(
    directory,
    formats=(".jpeg", ".jpg", ".png"),
    follow_links=True,
):
    """Make list of all files in the subdirs of `directory`, with their labels.
    Args:
      directory: The target directory (string).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".png").
    Returns:
      file_paths: list of file paths (strings).
    """
    subdirs = []
    for subdir in sorted(glob.glob(os.path.join(directory, "*"))):
        if os.path.isdir(os.path.join(directory, subdir)):
            subdirs.append(subdir)
    subdirs = [i for i in subdirs if not i.startswith(".")]

    # Build an index of the files
    # in the different class subfolders.
    pool = Pool()
    results = []
    filenames = []
    for dirpath in (subdir for subdir in subdirs):
        results.append(
            pool.apply_async(
                index_subdirectory, (dirpath, follow_links, formats)
            )
        )
    for res in results:
        partial_filenames = res.get()
        filenames += partial_filenames

    pool.close()
    pool.join()
    file_paths = [os.path.join(directory, fname) for fname in filenames]

    return file_paths


def iter_valid_files(directory, follow_links, formats):
    walk = os.walk(directory, followlinks=follow_links)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        if not os.path.split(root)[1].startswith("."):
            for fname in sorted(files):
                if fname.lower().endswith(formats):
                    yield root, fname


def index_subdirectory(directory, follow_links, formats):
    """Recursively walks directory and list image paths and their class index.
    Arguments:
      directory: string, target directory.
      follow_links: boolean, whether to recursively follow subdirectories
        (if False, we only list top-level images in `directory`).
      formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
    Returns:
      a list of relative file paths
        files.
    """
    dirname = os.path.basename(directory)
    valid_files = iter_valid_files(directory, follow_links, formats)
    filenames = []
    for root, fname in valid_files:
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)
    filenames_trim = [i for i in filenames if r"@" not in i]
    return filenames_trim
