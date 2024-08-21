import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# Constants
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 110
BOW_NUM_CLUSTERS = 40

# Initialize SIFT, FLANN, and BOW components
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
bow_kmeans_trainer = cv2.BOWKMeansTrainer(BOW_NUM_CLUSTERS)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)


# Function to get positive and negative image paths
def get_pos_and_neg_paths(i):
    pos_path = 'data/CarData/TrainImages/pos-%d.pgm' % (i + 1)
    neg_path = 'data/CarData/TrainImages/neg-%d.pgm' % (i + 1)
    return pos_path, neg_path


# Function to add a sample to the BOW trainer
def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)


# Add samples to the BOW trainer
for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)

# Cluster descriptors and set vocabulary for BOW extractor
voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

# EDA: Understanding the dataset
pos_images = []
neg_images = []

for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    if pos_img is not None and neg_img is not None:
        pos_images.append(pos_img)
        neg_images.append(neg_img)
print(f"Number of Positive Images: {len(pos_images)}")
print(f"Number of Negative Images: {len(neg_images)}")

# Check the size of images
print(f"Size of first positive image: {pos_images[0].shape}")
print(f"Size of first negative image: {neg_images[0].shape}")

# Visualize keypoints for a sample image

sample_image = pos_images[0]
keypoints, descriptors = sift.detectAndCompute(sample_image, None)
img_with_keypoints = cv2.drawKeypoints(sample_image, keypoints, None)
plt.imshow(img_with_keypoints, cmap='gray')
plt.show()

if descriptors is not None:
    pca = PCA(n_components=2)
    descriptors_2d = pca.fit_transform(descriptors)
    kmeans = KMeans(n_clusters=5, n_init=10)
    clusters = kmeans.fit_predict(descriptors_2d)
    # Plot with color coding for clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title('2D Visualization of SIFT Descriptors Using PCA (Color-Coded by Clusters)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
else:
    print("No Descriptors Detected")


# Function to extract BOW descriptors
def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)


# Histogram of Visual Words for EDA
def plot_histogram(descriptors, title):
    if descriptors is not None:
        # Convert descriptors to integer cluster indices
        cluster_indices = np.argmax(descriptors, axis=1)  # Assuming one descriptor per keypoint
        hist = np.bincount(cluster_indices, minlength=BOW_NUM_CLUSTERS)
        plt.bar(range(len(hist)), hist)
        plt.title(title)
        plt.show()
    else:
        print(f"No descriptors found for {title}")


# Visualize histogram for a positive and a negative image
pos_descriptors = extract_bow_descriptors(pos_images[0])
plot_histogram(pos_descriptors, "Histogram of Visual Words for Positive Image")

neg_descriptors = extract_bow_descriptors(neg_images[0])
plot_histogram(neg_descriptors, "Histogram of Visual Words for Negative Image")

# Training data preparation
training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_bow_descriptors(pos_img)
    if pos_descriptors is not None:
        training_data.extend(pos_descriptors)
        training_labels.append(1)
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.extend(neg_descriptors)
        training_labels.append(-1)

# Train SVM
svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))
# Cross-validation
# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Custom cross-validation loop
scores = []
for train_index, test_index in kf.split(training_data):
    X_train, X_test = np.array(training_data)[train_index], np.array(training_data)[test_index]
    y_train, y_test = np.array(training_labels)[train_index], np.array(training_labels)[test_index]

    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    # Predict on the test set
    _, y_pred = svm.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred.flatten() == y_test)
    scores.append(accuracy)

print(f"Custom cross-validation scores: {scores}")
# Test images
for test_img_path in ['data/CarData/TestImages/test-0.pgm', 'data/CarData/TestImages/test-1.pgm',
                      'data/car.jpg',
                      'data/haying.jpg',
                      'data/statue.jpg',
                      'data/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = extract_bow_descriptors(gray_img)
    prediction = svm.predict(descriptors)
    if prediction[1][0][0] == 1.0:
        text = 'car'
        color = (0, 255, 0)
    else:
        text = 'not car'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
