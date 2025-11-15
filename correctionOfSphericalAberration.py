import os
import statistics

import cv2
import numpy as np

def cartesian_image_to_polar(img):
    height, width = img.shape[:2]
    center = (width / 2, height/ 2)
    max_radius = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(img, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_image, center, max_radius

def polar_image_to_cartesian(img, center, max_radius):
    cartesian_image = cv2.linearPolar(img, center, max_radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    return cartesian_image


def edges_canny(image):
    return cv2.Canny(image, threshold1=50, threshold2=300)

def create_edge_list_from_image_list(img_list):
    edges_list = []
    for image in img_list:
        padded_img = pad_image(image)
        polar_image_pad, center_pad, max_radius_pad = cartesian_image_to_polar(padded_img)
        polar_image_from_canny_pad = edges_canny(polar_image_pad)
        edges_of_image = sharpness_of_radius(polar_image_from_canny_pad)
        edges_list.append(edges_of_image)

    return edges_list

def sharpness_of_radius(image):
    sharpness_array = []

    for x in range(0, image.shape[1], scan_size):
        edge_sum = []
        for i in range(x, x + scan_size):
            window = image[:, i:i + 1]

            edge_sum.append(np.sum(window))

        sharpness_array.append(statistics.median(edge_sum))

    return sharpness_array

def create_absolute_values(list_of_values):
    transposed = [list(col) for col in zip(*list_of_values)]
    list_of_high_values = []
    index = 0
    for row in transposed:
        high = 0
        for i, value in enumerate(row):
            if value > high:
                high = value
                index = i
        list_of_high_values.append((int(high), index))

    return list_of_high_values

def create_new_image(images_list, max_values):
    images = []
    for image in images_list:
        polar_image = cartesian_image_to_polar(image)[0]
        images.append(polar_image)

    columns  = []
    for x, values in enumerate(max_values):
        img = images[values[1]]
        window = img[:, x * scan_size:x * scan_size + scan_size]
        columns.append(window)

    new_img = np.hstack(columns)

    return new_img

def pad_image(img):
    h, w = img.shape[:2]
    pad_y, pad_x = int(2.5 * h), int(2.5 * w)

    img = cv2.copyMakeBorder(
        img,
        pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return img

def remove_pad(img):
    mask = img > 1

    cord = np.argwhere(mask)
    y0, x0 = cord.min(axis=0)
    y1, x1 = cord.max(axis=0)

    cropped = img[y0:y1, x0:x1]
    return cropped

def align_image(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    img1_f = np.float32(img1)
    img2_f = np.float32(img2)

    shift, response = cv2.phaseCorrelate(img1_f, img2_f)
    m = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    aligned = cv2.warpAffine(img2, m, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_NEAREST)

    return aligned

def create_image_from_polar(polar, ref_img):
    center, max_radius = cartesian_image_to_polar(ref_img)[1:]
    image = remove_pad(polar_image_to_cartesian(polar, center, max_radius))
    return image

def canny_sum(image):
    polar_image, center_pad, max_radius_pad = cartesian_image_to_polar(image)
    canny_from_polar_image = edges_canny(polar_image)
    total_edges = 0
    for x in range(0, canny_from_polar_image.shape[1], scan_size):
        window = canny_from_polar_image[:, x:x + scan_size]
        total_edges = total_edges + np.sum(window)

    return total_edges

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        images.append(folder + filename)
    return images

image_path_list = load_images_from_folder('./InputImagesSelectionSet6/')
scan_size = 15
aligned_images = []
img_for_alignment = ""

for i, img_path in enumerate(image_path_list):
    edges = canny_sum(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    print('edges of image: ',img_path , edges)
    if i == 0:
        aligned_images.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        img_for_alignment = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        aligned_images.append(align_image(img_for_alignment, cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)))


edge_list = create_edge_list_from_image_list(aligned_images)
absolut_values = create_absolute_values(edge_list)

for i, aligned_image in enumerate(aligned_images):
    aligned_images[i] = pad_image(aligned_image)

corrected_image = create_image_from_polar(create_new_image(aligned_images, absolut_values), aligned_images[0])

cv2.imshow('Corrected Image', corrected_image)
cv2.imwrite('corrected_image.jpeg', corrected_image)
print('edge count of corrected image', canny_sum(corrected_image))

cv2.waitKey(0)