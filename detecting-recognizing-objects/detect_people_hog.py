import cv2

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
OPENCV_MINOR_VERSION = int(cv2.__version__.split('.')[1])


def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and iy > oy and iy + ih < oy + oh


img = cv2.imread('data/haying.jpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)
if OPENCV_MAJOR_VERSION >= 5 or (OPENCV_MAJOR_VERSION == 4 and OPENCV_MINOR_VERSION >= 6):
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, groupThreshold=1.9)
else:
    found_rects, found_weights = hog.detectMultiScale(img, winStride=(4, 4), scale=1.02, finalThreshold=1.9)
print(found_rects)
print(found_weights)
# Non-maximum suppression (NMS)
found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])
# drawing bounding boxes around detected objects and labeling
for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Women in Hayfield Detected', img)
cv2.waitKey(0)
