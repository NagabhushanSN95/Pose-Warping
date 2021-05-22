# Shree KRISHNAya Namaha
# Warps frame using pose info
# Author: Nagabhushan S N
# Last Modified: 05/11/2020

import datetime
import time
import traceback
from pathlib import Path

import numpy
import skimage.io

import Imath
import OpenEXR


def camera_intrinsic_transform_05(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
    start_y, start_x = patch_start_point
    camera_intrinsics = numpy.eye(4)
    camera_intrinsics[0, 0] = 2100
    camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
    camera_intrinsics[1, 1] = 2100
    camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
    return camera_intrinsics


def read_image(path: Path) -> numpy.ndarray:
    image = skimage.io.imread(path.as_posix())
    return image


def read_depth(path: Path) -> numpy.ndarray:
    if path.suffix == '.png':
        depth = skimage.io.imread(path.as_posix())
    elif path.suffix == '.npy':
        depth = numpy.load(path.as_posix())
    elif path.suffix == '.npz':
        with numpy.load(path.as_posix()) as depth_data:
            depth = depth_data['arr_0']
    elif path.suffix == '.exr':
        exr_file = OpenEXR.InputFile(path.as_posix())
        raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
        depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
        height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
        width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
        depth = numpy.reshape(depth_vector, (height, width))
    else:
        raise RuntimeError(f'Unknown depth format: {path.suffix}')
    return depth


def forward_warp(frame1: numpy.ndarray, depth1: numpy.ndarray, intrinsic: numpy.ndarray,
                 transformation1: numpy.ndarray, transformation2: numpy.ndarray, is_image: bool = False):
    """
    Warps the frame
    @return: warped_frame2, mask, trans_pos.
    """
    h, w, c = frame1.shape
    assert depth1.shape == (h, w)
    transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

    y1d = numpy.array(range(h))
    x1d = numpy.array(range(w))
    x2d, y2d = numpy.meshgrid(x1d, y1d)
    ones_2d = numpy.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

    intrinsic_inv = numpy.linalg.inv(intrinsic)
    intrinsic_4d = intrinsic[None, None]
    intrinsic_inv_4d = intrinsic_inv[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]

    unnormalized_pos = numpy.matmul(intrinsic_inv_4d, pos_vectors_homo)
    unit_pos_vecs = unnormalized_pos / numpy.linalg.norm(unnormalized_pos, axis=2, keepdims=True)
    world_points = depth_4d * unit_pos_vecs
    world_points_homo = numpy.concatenate([world_points, ones_4d], axis=2)
    trans_world_homo = numpy.matmul(trans_4d, world_points_homo)
    trans_world = trans_world_homo[:, :, :3]
    trans_norm_points = numpy.matmul(intrinsic_4d, trans_world)
    trans_pos = trans_norm_points[:, :, :2, 0] / trans_norm_points[:, :, 2:3, 0]  # transformed positions

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
    trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

    weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]

    sat_depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
    log_depth1 = numpy.log(1 + sat_depth1)
    depth_weights = numpy.exp(log_depth1 / log_depth1.max() * 50)
    depth_weights_3d = depth_weights[:, :, None]

    warped_image = numpy.zeros(shape=(h + 2, w + 2, c), dtype=numpy.float32)
    warped_weights = numpy.zeros(shape=(h + 2, w + 2), dtype=numpy.float32)

    numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]),
                 frame1 * weight_nw_3d / depth_weights_3d)
    numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]),
                 frame1 * weight_sw_3d / depth_weights_3d)
    numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]),
                 frame1 * weight_ne_3d / depth_weights_3d)
    numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]),
                 frame1 * weight_se_3d / depth_weights_3d)

    numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw / depth_weights)
    numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw / depth_weights)
    numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne / depth_weights)
    numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se / depth_weights)

    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]

    mask = cropped_weights > 0
    with numpy.errstate(invalid='ignore'):
        final_image = numpy.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

    if is_image:
        clipped_image = numpy.clip(final_image, a_min=0, a_max=255)
        final_image = numpy.round(clipped_image).astype('uint8')
    return final_image, mask, trans_pos


def demo1():
    frame1_path = Path('../Data/frame1.png')
    frame2_path = Path('../Data/frame2.png')
    depth1_path = Path('../Data/depth1.exr')
    transformation1 = numpy.eye(4)
    transformation2 = numpy.eye(4)

    frame1 = read_image(frame1_path)
    frame2 = read_image(frame2_path)
    depth1 = read_depth(depth1_path)
    intrinsic = camera_intrinsic_transform_05()

    warped_frame2 = forward_warp(frame1, depth1, intrinsic, transformation1, transformation2)[0]
    skimage.io.imsave('frame1.png', frame1)
    skimage.io.imsave('frame2.png', frame2)
    skimage.io.imsave('frame2_warped.png', warped_frame2)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
