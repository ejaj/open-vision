import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the query image.
folder = 'data/tattoos'
query = cv2.imread(os.path.join(folder, 'query.png'), cv2.IMREAD_GRAYSCALE)

# Initialize lists to store file names, images, and descriptors
files = []
descriptors = []

# Walk through the folder to find all .npy files
for (dirpath, dirnames, filenames) in os.walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith('npy') and f != 'query.npy':
            descriptors.append(f)

# Create the SIFT detector.
sift = cv2.SIFT_create()

# Perform SIFT feature detection and description on the query image.
query_kp, query_ds = sift.detectAndCompute(query, None)

# Define FLANN-based matching parameters.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Create the FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Define the minimum number of good matches for a suspect.
MIN_NUM_GOOD_MATCHES = 10

greatest_num_good_matches = 0
prime_suspect = None
prime_suspect_img = None
prime_suspect_kp = None
prime_suspect_matches = None

print('>> Initiating picture scan...')

for d in descriptors:
    print('--------- analyzing %s for matches ------------' % d)
    suspect_ds = np.load(os.path.join(folder, d))
    matches = flann.knnMatch(query_ds, suspect_ds, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    num_good_matches = len(good_matches)
    name = d.replace('.npy', '').upper()

    # Load the suspect image for potential display later
    suspect_img = cv2.imread(os.path.join(folder, name.lower() + '.png'), cv2.IMREAD_GRAYSCALE)
    suspect_kp = sift.detect(suspect_img, None)

    if num_good_matches >= MIN_NUM_GOOD_MATCHES:
        print('%s is a suspect! (%d matches)' % (name, num_good_matches))
        if num_good_matches > greatest_num_good_matches:
            greatest_num_good_matches = num_good_matches
            prime_suspect = name
            prime_suspect_img = suspect_img
            prime_suspect_kp = suspect_kp
            prime_suspect_matches = good_matches
    else:
        print('%s is NOT a suspect. (%d matches)' % (name, num_good_matches))

if prime_suspect is not None:
    print('Prime suspect is %s.' % prime_suspect)

    # Draw the matches between the query image and the prime suspect image
    img_matches = cv2.drawMatches(query, query_kp, prime_suspect_img, prime_suspect_kp, prime_suspect_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the image
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title(f'Matches between query and {prime_suspect}')
    plt.show()

else:
    print('There is no suspect.')
