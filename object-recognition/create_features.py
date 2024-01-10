import os
import argparse
import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans


class DenseDetector():
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints


class SIFTExtractor():
    def __init__(self):
        self.extractor = cv2.SIFT_create()

    def compute(self, image, kps):
        if image is None:
            raise TypeError("Not a valid image")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self.extractor.detectAndCompute(gray_image, None)
        return kps, des


class Quantizer(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def quantize(self, datapoints):
        kmeans = KMeans(self.num_clusters, n_init=max(self.num_retries, 1), max_iter=10, tol=1.0)
        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)
        return input_data / sum_input if sum_input > 0 else input_data

    def get_feature_vector(self, img, kmeans, centroids):
        kps = DenseDetector().detect(img)
        kps, fvs = self.extractor.compute(img, kps)

        # Check if descriptors are empty
        if fvs is None or len(fvs) == 0:
            print("No descriptors found for image.")
            return None

        # Convert feature vectors to float32
        fvs = fvs.astype(np.float32)

        # Debugging: Check data type and shape of feature vectors
        print("Feature vectors dtype:", fvs.dtype, "Shape:", fvs.shape)

        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)
        for i, _ in enumerate(fvs):
            fv[labels[i]] += 1
        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalize(fv_image)

class FeatureExtractor(object):
    def extract_image_features(self, img):
        kps = DenseDetector().detect(img)
        kps, fvs = SIFTExtractor().compute(img, kps)
        return fvs

    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []
        cur_label = ''
        for item in input_map:
            if item['label'] != cur_label:
                count = 0
                cur_label = item['label']
            count += 1
            if count > num_samples_to_fit:
                continue
            img = cv2.imread(item['image'])
            img = resize_to_size(img, 150)
            fvs = self.extract_image_features(img)
            kps_all.extend(fvs)
        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Creates features for given images')
    parser.add_argument("--samples", dest="cls", nargs="+", action="append", required=True,
                        help="Folders containing the training images. The first element needs to be the class label.")
    parser.add_argument("--codebook-file", dest='codebook_file', required=True,
                        help="Base file name to store the codebook")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
                        help="Base file name to store the feature map")
    return parser


def load_input_map(label, input_folder):
    combined_data = []
    if not os.path.isdir(input_folder):
        raise IOError(f"The folder {input_folder} doesn't exist")
    for root, _, files in os.walk(input_folder):
        for filename in [x for x in files if x.endswith('.jpg')]:
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
    return combined_data


def extract_feature_map(input_map, kmeans, centroids):
    feature_map = []
    for item in input_map:
        print("Extracting features for", item['image'])
        img = cv2.imread(item['image'])
        img = resize_to_size(img, 150)
        feature_vector = FeatureExtractor().get_feature_vector(img, kmeans, centroids)
        if feature_vector is not None:
            feature_map.append({'label': item['label'], 'feature_vector': feature_vector})
    return feature_map


def resize_to_size(input_image, new_size=150):
    h, w = input_image.shape[:2]
    ds_factor = new_size / float(h) if h > w else new_size / float(w)
    new_size = (int(w * ds_factor), int(h * ds_factor))
    return cv2.resize(input_image, new_size)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_map = []
    for cls in args.cls:
        assert len(cls) >= 2, "Format for classes is `<label> file`"
        label, folder = cls[0], cls[1]
        input_map += load_input_map(label, folder)

    print("===== Building codebook =====")
    kmeans, centroids = FeatureExtractor().get_centroids(input_map)
    if args.codebook_file:
        with open(args.codebook_file, 'wb') as f:
            pickle.dump((kmeans, centroids), f)

    print("===== Building feature map =====")
    feature_map = extract_feature_map(input_map, kmeans, centroids)
    if args.feature_map_file:
        with open(args.feature_map_file, 'wb') as f:
            pickle.dump(feature_map, f)
