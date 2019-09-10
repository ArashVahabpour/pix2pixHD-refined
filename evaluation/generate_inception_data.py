import numpy as np
import os
import cv2 as cv
import glob
# import matplotlib.pyplot as plt
from string import ascii_uppercase


# def get_edges(input_image):
#     out_edges = np.zeros(input_image.shape, dtype=np.bool)
#     out_edges[:,1:] |= (input_image[:,1:] != input_image[:,:-1])
#     out_edges[:,:-1] |= (input_image[:,1:] != input_image[:,:-1])
#     out_edges[1:,:] |= (input_image[1:,:] != input_image[:-1,:])
#     out_edges[:-1,:] |= (input_image[1:,:] != input_image[:-1,:])
#     out_edges &= cv.erode((input_image != 0).astype(np.uint8) * 255, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))).astype(np.bool)  # in addition to pix2pixHD logic
#     return out_edges


def crop_and_resize(input_image, bg, x1, y1, w, h, interpolation, dtype=np.uint8):
    """
    Args:
        input_image: image to be cropped
        bg: output background color 0 or 255, etc.

    Output:
        a 256x256 cropped and resized image
    """
    ratio = 250. / max(w, h)
    crop = input_image[y1:y1 + h, x1:x1 + w]
    crop_resized = cv.resize(crop, (int(w * ratio), int(h * ratio)), interpolation=interpolation)
    shape = crop_resized.shape
    output_image = np.ones((256, 256, 3) if len(input_image.shape) == 3 else (256, 256), dtype=dtype) * bg
    output_image[128 - shape[0] // 2:128 - shape[0] // 2 + shape[0],
    128 - shape[1] // 2:128 - shape[1] // 2 + shape[1]] = crop_resized
    return output_image


# def stitch_back(canvas_, _256x256, x1, y1, w, h):
#     resized_w, resized_h = int(250. * w / max(w, h)), int(250. * h / max(w, h))
#     crop = _256x256[128-resized_h//2:128-resized_h//2+resized_h, 128-resized_w//2:128-resized_w//2+resized_w]
#     crop_orig_size = cv.resize(crop, (w, h))
#     canvas_[y1:y1+h, x1:x1+w] *= (crop_orig_size == 255)
#     crop_orig_size[crop_orig_size == 255] = 0
#     canvas_[y1:y1+h, x1:x1+w] += crop_orig_size
#     return canvas_


def mkdir(path):
    if os.path.exists(path):
        print('Warning: old directory exists')
    else:
        os.mkdir(path)


# def histEQ(img, binerized_image):
#     binerized_image = np.array(binerized_image // 255, dtype=bool)
#     hist, bins = np.histogram(img[binerized_image].flatten(), 256, [0, 256])
#     output = hist / (img[binerized_image].flatten()).shape[0]
#     CSum = output.cumsum(axis=0)
#     return CSum[img] * 255

def create_inception_dataset(cityscapes_dir, results_dir, destination_dir, separate_biggest=True, separate_two_biggest=True):
    mkdir(destination_dir)

    # subset in ['train', 'test']:#'train', 'test':
    subset = 'test'

        # for id in range(35):
    # id = 26

    for crop_type in ['ground_truth', 'stage_2', 'stage_4']:
        mkdir(destination_dir + str(crop_type))

    # for letter in ascii_uppercase[:5]:
    #     mkdir(destination_dir + str(id) + '/' + subset + '_' + letter)

    for image_filename in glob.glob(cityscapes_dir + subset + '_img/*.png'):
        print('Processing: ' + os.path.basename(image_filename))


        image = cv.imread(image_filename)
        inst = cv.imread(image_filename.replace('leftImg8bit', 'gtFine_instanceIds').replace('_img/', '_inst/'), cv.IMREAD_ANYDEPTH)[:, :, None]  # WARNING: NOT DIRECTLY COMPATIBLE WITH UINT8 FORMAT
        label = cv.imread(image_filename.replace('leftImg8bit', 'gtFine_labelIds').replace('_img/', '_label/'))


        # encoding input images to appropriate format
        label = label[:, :, 0]
        objs_mask = label == id

        _, cc = cv.connectedComponents(objs_mask.astype('uint8'))

        # seperate each object (e.g. car) out
        obj_mask_list = [np.ndarray.astype((cc == i), np.uint8) * 255 for i in range(1, len(np.unique(cc)))]
        obj_mask_biggest_list = []
        obj_mask_two_biggest_list = []

        def select_biggest(candidate_object_ids):
            """
            Find the biggest mask given a list of candidate object ids
            """
            biggest_instid = candidate_object_ids[np.argmax([(inst == c).sum() for c in candidate_object_ids])]
            biggest = np.ndarray.astype(inst.squeeze() == biggest_instid, np.uint8) * 255

            return biggest

        if separate_biggest:
            for obj_mask in obj_mask_list:
                uniques = np.unique(obj_mask[:, :, None] // 255 * inst)[1:]
                if len(uniques) > 1:  # if a stack of cars
                    obj_mask_biggest = select_biggest(uniques)
                    obj_mask_biggest_list.append(obj_mask_biggest)

        if separate_two_biggest:
            for obj_mask_biggest in obj_mask_biggest_list:
                try:
                    obj_mask_biggest_dilated = cv.dilate(obj_mask_biggest.astype(np.uint8),
                                                         kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
                    biggest_instid = list(set(np.unique(obj_mask_biggest[:,:,None]//255 * inst)) - {0})
                    assert len(biggest_instid) == 1, 'Unexpected instance map in {}'.format(image_filename)
                    biggest_instid = biggest_instid[0]
                    neighbor_ids = set(np.unique(obj_mask_biggest_dilated[:,:,None]//255 * inst)) - {0, biggest_instid}
                    if len(neighbor_ids) == 2:  # then the stack itself has two cars, so no need
                        continue
                    neighbor_ids = filter(lambda x: id <= x//1000 < id+1, neighbor_ids)  # select only the neighbor objects of the correct type, e.g. cars

                    obj_mask_two_biggest = select_biggest(list(neighbor_ids)) | obj_mask_biggest
                    if not any([np.array_equal(obj_mask_two_biggest, obj_mask) for obj_mask in obj_mask_list]):
                        obj_mask_two_biggest_list.append(obj_mask_two_biggest)

                except Exception as e:
                    print(e)


        for i, (obj_mask, mask_type) in enumerate(zip(obj_mask_list + obj_mask_biggest_list + obj_mask_two_biggest_list,
                                                      ['stack'] * len(obj_mask_list) +
                                                      ['biggest'] * len(obj_mask_biggest_list) +
                                                      ['twobiggest'] * len(obj_mask_two_biggest_list))):

            try:
                contours = cv.findContours(obj_mask, cv.RETR_EXTERNAL, 2)[0]
            except ValueError:
                continue

            cnt = contours[0]
            x1, y1, w, h = cv.boundingRect(cnt)

            if max(w, h) < 256:
                continue  # skip small objects

            # 3 white canvases, 2 black canvases
            canvas = [np.ones((256, 256), dtype=np.uint8) * 255 for _ in range(3)] + \
                     [np.zeros((256, 256), dtype=np.uint8) * 255 for _ in range(2)]

            obj_tmp = np.tile(255 - obj_mask[:,:, None], (1,1,3))
            obj_tmp += image * (obj_mask[:,:, None]//255)


            obj = crop_and_resize(obj_tmp, 255, x1, y1, w, h, cv.INTER_LINEAR)
            obj_mask_cropped_resized = crop_and_resize(obj_mask//255 * inst[:,:,0], 0, x1, y1, w, h, cv.INTER_NEAREST, dtype=np.uint16)

            canvas[0] = ((obj_mask_cropped_resized != 0) & ~get_edges(obj_mask_cropped_resized)).astype(np.uint8) * 255
            canvas[1] = obj
            canvas[2] = 255 - cv.Canny(obj, 50, 150)

            ratio = 256. / image.shape[1]
            all_cars_map = cv.resize((label == 26).astype(np.uint8) * 255,
                                     None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            single_car_map = cv.resize(obj_mask[:, :, None].astype(np.uint8),
                                       None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            cmr_h = all_cars_map.shape[0]
            canvas[3][128 - cmr_h // 2:128 - cmr_h // 2 + cmr_h, :] = all_cars_map
            canvas[4][128 - cmr_h // 2:128 - cmr_h // 2 + cmr_h, :] = single_car_map

            if grayscale:
                canvas[1] = cv.cvtColor(canvas[1], cv.COLOR_BGR2GRAY)
                canvas[1] = histEQ(canvas[1],canvas[0])
            for letter, writable in zip(list(str(ascii_uppercase[:5])), canvas):
                output_filename = destination_dir + str(id) + '/' + subset + '_' + letter + '/' \
                           + '_'.join([os.path.splitext(os.path.basename(image_filename))[0].replace('_leftImg8bit', '')]
                                      + [str(q) for q in [x1, y1, w, h]])\
                           + '_' + mask_type + '.png'
                cv.imwrite(output_filename, writable)

                # if flip:
                #     cv.imwrite(output_filename.replace(os.path.basename(output_filename), 'flip_' + os.path.basename(output_filename)),
                #                cv.flip(writable, 1))


# def join_all(cityscapes_dir, objs_dir, generated_grayscale_dir, destination_dir):
#     """
#
#     :param cityscapes_dir:
#     :param objs_dir:
#     :param generated_grayscale_dir: test results of the refinement network on objects
#     :param destination_dir:
#     :return:
#     """
#     white_canvas = np.ones((1024, 2048, 3), np.uint8) * 255
#
#     mkdir(destination_dir)
#     for subset in ['train']:#''train', 'test':
#         for image_filename in glob.glob(cityscapes_dir + subset + '_img/*.png'):
#             if 'bremen_000122_000019' not in image_filename:
#                 continue
#             print('THE LINE ABOVE TO BE REMOVED')
#             # try:
#             if True:
#                 canvas = white_canvas.copy()
#                 # for id in range(35):
#                 id = 26
#
#                 obj_filename_list = glob.glob('{}{}/{}_B/{}*'.format(objs_dir, id, subset, os.path.basename(image_filename).split('.')[0].replace('leftImg8bit', '')))
#                 obj_filename_list = list(filter(lambda x: 'biggest' not in x, obj_filename_list)) + list(filter(lambda x: 'biggest' in x, obj_filename_list))  # process stack of objects first
#
#                 for obj_filename in obj_filename_list:
#                     x1, y1, w, h, is_biggest = os.path.basename(obj_filename).split('.')[0].split('_')[-5:]
#                     x1, y1, w, h = (int(s) for s in (x1, y1, w, h))
#                     is_biggest = bool(is_biggest)
#
#                     obj = cv.imread(generated_grayscale_dir + os.path.basename(obj_filename).replace('.png', '_synthesized_image2.png'))[:256, :]
#
#                     stitch_back(canvas, obj, x1, y1, w, h)
#
#                 output_filename = '{}/{}'.format(destination_dir, os.path.basename(image_filename))
#                 print(os.path.basename(image_filename))
#                 cv.imwrite(output_filename, 255 - cv.Canny(cv.resize(canvas, (0,0), fx=0.5, fy=0.5), 50, 150))
#
#             # except:
#             #     print('Failed: ' + os.path.basename(image_filename))


def main():
    create_dataset(cityscapes_dir='/home/shared/datasets/cityscapes.pix2pixHD.folders/',
                   results_dir='',
                   destination_dir='/home/shared/datasets/GAN_project/inception')
    # join_all('/home/shared/datasets/cityscapes.pix2pixHD.folders/',
    #          '/home/shared/datasets/cityscapes_objects/',
    #          '/home/arash/Desktop/pix2pixHD-edge-grayscale-back2back/results/cars/test_100/images/',
    #          '/home/arash/Desktop/cityscapes_edge_maps.train')



main()
