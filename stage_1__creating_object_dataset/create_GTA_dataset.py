import cv2 as cv
import numpy as np
import glob
import os
import pickle
from string import ascii_uppercase


class Rectangle:
    def __init__(self, y1, x1, y2, x2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


def cross_verify_mask(ground_truth_mask, detectron_mask, threshold):
    """
    Cross verify ground truth mask with detectron's masks and declare if there is enough overlap

    Args:
        ground_truth_mask: True / False np array of shape (h, w)
        detectron_mask: OpenCV contour of shape (None, 2)
        threshold: minimum amount of overlap

    Returns: True / False

    """
    IoU = lambda x, y: np.sum(cv.bitwise_and(x, y)) * 1. / np.sum(cv.bitwise_or(x, y))

    return IoU(ground_truth_mask, detectron_mask) >= threshold


def find_all_good_masks(ground_truth_label_map, detectron_contours, threshold):
    """
    Generate all masks that can be verified with detectron
    """

    g_masks = [np.tile(np.ndarray.astype((ground_truth_label_map == i)[:, :, None], np.uint8) * 255, (1, 1, 3))
               for i in range(1, len(np.unique(ground_truth_label_map)))]

    d_masks = []
    for c in detectron_contours:
        try:
            d_mask = np.zeros(list(ground_truth_label_map.shape) + [3], np.uint8)
            cv.drawContours(d_mask, [c], -1, (255, 255, 255), -1)
            d_masks.append(d_mask)
        except Exception:
            print(c.shape)  # should print (2,) corresponding to an empty contour

    good_masks = []
    for g_mask in g_masks:
        for d_mask in d_masks:
            if cross_verify_mask(g_mask, d_mask, threshold):
                good_masks.append(g_mask)
                break

    return good_masks


# Read the image and the label map

dataset_dir = '/media/qiujing/e5439522-63c4-4b7c-a968-fefee6a3d960/arash/GTA'
assert os.path.exists(dataset_dir), 'Dataset path is wrong. Please modify.'

# create output dirs
def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)

mkdir(dataset_dir + '/cars_dataset')
for subset in ['train', 'test']:
    for letter in ascii_uppercase[:5]:
        mkdir(dataset_dir + '/cars_dataset/{}_{}'.format(subset, letter))


np.random.seed(0)
which_subset = np.random.rand(1000) > 0.05  # train/test selection



output_idx = 0

for image_dir in glob.glob(dataset_dir + '/images/*.png'):

    image_id = os.path.splitext(os.path.basename(image_dir))[0]
    # print(image_id)
    label_dir = dataset_dir + '/labels/{}.png'.format(image_id)
    contours_dir = dataset_dir + '/contours/{}.pkl'.format(image_id)

    if not os.path.exists(contours_dir):
        # print('Contours not pickled for this image.')
        continue

    L = cv.imread(label_dir)
    I = cv.imread(image_dir)

    # Create cars mask

    car_color = np.array([142, 0, 0])

    cars_mask = np.all(L == car_color.reshape(1, 1, -1), axis=2)
    cars_mask_uint8 = np.ndarray.astype(cars_mask, np.uint8)

    # Find connected components (individual cars)
    ret, car_labels = cv.connectedComponents(cars_mask_uint8)

    with open(contours_dir, "rb") as openfile:
        d_contours = pickle.load(openfile, encoding='iso-8859-1')
    d_contours = [d_contour.squeeze() for d_contour in d_contours]

    all_good_masks = find_all_good_masks(car_labels, d_contours, 0.7)
    # print(len(all_good_masks))
    for good_mask in all_good_masks:
        car_masked = good_mask * I
        mask_uint8 = np.ndarray.astype(good_mask, np.uint8)

        # Separating out the desired car
        mask_pixels = np.argwhere(np.squeeze(good_mask[:, :, 0]) == 255)  # car's pixels

        r = Rectangle(*list(mask_pixels.min(0)) + list(mask_pixels.max(0)))  # car's bounding rectangle

        if max(r.width, r.height) <= 256:  # discard small cars
            # print('Car is too small')
            continue

        car_cropped = car_masked[r.y1:r.y2, r.x1:r.x2]
        mask_cropped = mask_uint8[r.y1:r.y2, r.x1:r.x2]

        ratio = 250. / max(r.width, r.height)  # a few pixels less than 256 to avoid boundary pixels

        car_cropped_resized = cv.resize(car_cropped, (int(r.width * ratio), int(r.height * ratio)),
                                        interpolation=cv.INTER_AREA)
        mask_cropped_resized = cv.resize(mask_cropped, (int(r.width * ratio), int(r.height * ratio)),
                                         interpolation=cv.INTER_NEAREST)
        import matplotlib.pyplot as plt

        # Creating the car's edge data
        canvas = [np.zeros((256, 256), np.uint8) for _ in range(5)]

        h, w = car_cropped_resized.shape[:2]

        # input
        canvas[0][128 - h // 2:128 - h // 2 + h, 128 - w // 2:128 - w // 2 + w] = mask_cropped_resized[:, :, 0].squeeze()

        # output
        car_padded = cv.copyMakeBorder(car_cropped_resized, 1, 1, 1, 1, cv.BORDER_CONSTANT,
                                       value=[0, 0, 0])  # add padding to improve canny on boundary pixels
        car_padded = cv.cvtColor(car_padded, cv.COLOR_BGR2GRAY)
        canvas[1][128 - h // 2 - 1: 128 + h - h // 2 + 1, 128 - w // 2 - 1: 128 + w - w // 2 + 1] = car_padded
        canvas[1] = 255 - canvas[1]

        # canny output
        canvas[2] = 255 - cv.Canny(canvas[1], 50, 150)

        # context input
        ratio = 256./I.shape[1]
        all_cars_map = cv.resize(cars_mask.astype(np.uint8) * 255,
                                 None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
        single_car_map = cv.resize(good_mask[:, :, 0].astype(np.uint8),
                                   None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)

        cmr_h = all_cars_map.shape[0]
        canvas[3][128 - cmr_h//2:128 - cmr_h//2 + cmr_h, :] = all_cars_map
        canvas[4][128 - cmr_h//2:128 - cmr_h//2 + cmr_h, :] = single_car_map


        # Save output data
        for i, (letter, writable) in enumerate(zip(list(str(ascii_uppercase[:5])), canvas)):
            cv.imwrite(dataset_dir + '/cars_dataset/{}_{}/{:06d}.png'.format('train' if which_subset[output_idx % len(which_subset)] else 'test', letter, output_idx), canvas[i])

        print('Image {}'.format(os.path.basename(image_dir)), output_idx)

        output_idx += 1
