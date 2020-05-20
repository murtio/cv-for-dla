import numpy as np
import cv2
import os


cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
IMG = './data_BCEV2/0/jpg/'

localizations_dir = './data_BCEV2/0/xml/'
localaizations_sorted_files = sorted(os.listdir(localizations_dir))


def load_images():  # load the images of the file
    images = []
    for file in sorted(os.listdir(IMG)):
        if file.endswith('.jpg'):
            # images.append(cv2.imread(IMG + file))
            yield cv2.imread(IMG + file)

    # return images


def laba_alg(im):
    # Image area
    area = im.shape[0] * im.shape[1]
    # 1. Grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 2. Otsu
    ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # 3. Erosion; in LABA there's not details of the kernel size
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(thr, kernel, iterations=1)
    # 4. Find contours
    image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 5. Bounding box
    # 6. ROI - remove contours with area < 60A and less than 4
    rects_lst = []
    merg_lst = []
    for c in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[c])
        if w * h > area * 0.6 or w * h <= 4:
            continue
        rects_lst.append((w * h, x, y, w, h))
    # 7. merge two boxes if they are contained
    rects_lst.sort(key=lambda tup: tup[0])
    for i in range(len(rects_lst)):
        if i > len(rects_lst):
            break
        cnt_x = int(rects_lst[i][1] + (rects_lst[i][3] / 2))
        cnt_y = int(rects_lst[i][2] + (rects_lst[i][4] / 2))
        for j in range(len(rects_lst)):
            if i == j:
                continue
            try:
                temp_range_x = range(rects_lst[j][1], rects_lst[j][1] + rects_lst[j][3])
                temp_range_y = range(rects_lst[j][2], rects_lst[j][2] + rects_lst[j][4])
                if cnt_x in temp_range_x and cnt_y in temp_range_y:
                    if rects_lst[i][0] < rects_lst[j][0]:
                        merg_lst.append(rects_lst[i])
            except IndexError:
                continue
    # draw bounding boxes
    filtered_lst = [x[1:] for x in rects_lst if x not in merg_lst]
    for rec in filtered_lst:
        x, y, w, h = rec
        cv2.rectangle(thr, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return thr, filtered_lst


for img in load_images():
    # call LABA method
    laba_img, laba_coords = laba_alg(img)
    # call XY-cut

    # call Docscrum
    # call mask-rcnn

    # resizing only for the purpose of display; since the xml files coords are based on the original size.
    resize = cv2.resize(laba_img, (int(img.shape[1] * 0.45), int(img.shape[0] * 0.45)))
    cv2.imshow('Original', resize)
    cv2.waitKey(0)
