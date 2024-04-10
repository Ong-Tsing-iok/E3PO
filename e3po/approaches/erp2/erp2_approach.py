# E3PO, an open platform for 360˚ video streaming simulation and evaluation.
# Copyright 2023 ByteDance Ltd. and/or its affiliates
#
# This file is part of E3PO.
#
# E3PO is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# E3PO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see:
#    <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>

import os
import cv2
import copy
import yaml
import shutil
import numpy as np
from e3po.utils import get_logger
from e3po.utils.data_utilities import transcode_video, segment_video, resize_video
from e3po.utils.decision_utilities import predict_motion_tile
from e3po.utils.projection_utilities import fov_to_3d_polar_coord, \
    _3d_polar_coord_to_pixel_coord


def video_analysis(user_data, video_info):
    """
    This API allows users to analyze the full 360 video (if necessary) before the pre-processing starts.
    Parameters
    ----------
    user_data: is initially set to an empy object and users can change it to any structure they need.
    video_info: is a dictionary containing the required video information.

    Returns
    -------
    user_data:
        user should return the modified (or unmodified) user_data as the return value.
        Failing to do so will result in the loss of the information stored in the user_data object.
    """

    user_data = user_data or {}
    user_data["video_analysis"] = []

    return user_data


def init_user(user_data, video_info):
    """
    Initialization function, users initialize their parameters based on the content passed by E3PO

    Parameters
    ----------
    user_data: None
        the initialized user_data is none, where user can store their parameters
    video_info: dict
        video information of original video, user can perform preprocessing according to their requirement

    Returns
    -------
    user_data: dict
        the updated user_data
    """

    user_data = user_data or {}
    user_data["video_info"] = video_info
    user_data["config_params"] = read_config()
    user_data["chunk_idx"] = -1

    return user_data


def read_config():
    """
    read the user-customized configuration file as needed

    Returns
    -------
    config_params: dict
        the corresponding config parameters
    """

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/erp.yml"
    with open(config_path, 'r', encoding='UTF-8') as f:
        opt = yaml.safe_load(f.read())['approach_settings']

    background_flag = opt['background']['background_flag']
    converted_height = opt['video']['converted']['height']
    converted_width = opt['video']['converted']['width']
    background_height = opt['background']['height']
    background_width = opt['background']['width']
    tile_height_num = opt['video']['tile_height_num']
    tile_width_num = opt['video']['tile_width_num']
    total_tile_num = tile_height_num * tile_width_num
    tile_width = int(opt['video']['converted']['width'] / tile_width_num)
    tile_height = int(opt['video']['converted']['height'] / tile_height_num)
    if background_flag:
        background_info = {
            "width": opt['background']['width'],
            "height": opt['background']['height'],
            "tile_width": int(opt['background']['width'] / tile_width_num),
            "tile_height": int(opt['background']['height'] / tile_height_num),
            "background_projection_mode": opt['background']['projection_mode']
        }
    else:
        background_info = {}

    motion_history_size = opt['video']['hw_size'] * 1000
    motino_prediction_size = opt['video']['pw_size']
    ffmpeg_settings = opt['ffmpeg']
    if not ffmpeg_settings['ffmpeg_path']:
        assert shutil.which('ffmpeg'), '[error] ffmpeg doesn\'t exist'
        ffmpeg_settings['ffmpeg_path'] = shutil.which('ffmpeg')
    else:
        assert os.path.exists(ffmpeg_settings['ffmpeg_path']), \
            f'[error] {ffmpeg_settings["ffmpeg_path"]} doesn\'t exist'
    projection_mode = opt['approach']['projection_mode']
    converted_projection_mode = opt['video']['converted']['projection_mode']

    config_params = {
        "background_flag": background_flag,
        "converted_height": converted_height,
        "converted_width": converted_width,
        "background_height": background_height,
        "background_width": background_width,
        "tile_height_num": tile_height_num,
        "tile_width_num": tile_width_num,
        "total_tile_num": total_tile_num,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "background_info": background_info,
        "motion_history_size": motion_history_size,
        "motion_prediction_size": motino_prediction_size,
        "ffmpeg_settings": ffmpeg_settings,
        "projection_mode": projection_mode,
        "converted_projection_mode": converted_projection_mode
    }

    return config_params


def preprocess_video(source_video_uri, dst_video_folder, chunk_info, user_data, video_info):
    """
    Self defined preprocessing strategy

    Parameters
    ----------
    source_video_uri: str
        the video uri of source video
    dst_video_folder: str
        the folder to store processed video
    chunk_info: dict
        chunk information
    user_data: dict
        store user-related parameters along with their required content
    video_info: dict
        store video information

    Returns
    -------
    user_video_spec: dict
        a dictionary storing user specific information for the preprocessed video
    user_data: dict
        updated user_data
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    # update related information
    if user_data['chunk_idx'] == -1:
        user_data['chunk_idx'] = chunk_info['chunk_idx']
        user_data['tile_idx'] = 0
        user_data['transcode_video_uri'] = source_video_uri
        user_data['generating_background'] = False
    else:
        if user_data['chunk_idx'] != chunk_info['chunk_idx']:
            user_data['chunk_idx'] = chunk_info['chunk_idx']
            user_data['tile_idx'] = 0
            user_data['transcode_video_uri'] = source_video_uri
            user_data['generating_background'] = False
            

    # transcoding
    src_projection = config_params['projection_mode']
    dst_projection = config_params['converted_projection_mode']
    if src_projection != dst_projection and user_data['tile_idx'] == 0:
        src_resolution = [video_info['height'],video_info['width']]
        dst_resolution = [config_params['converted_height'], config_params['converted_width']]
        user_data['transcode_video_uri'] = transcode_video(
            source_video_uri, src_projection, dst_projection, src_resolution, dst_resolution,
            dst_video_folder, chunk_info, config_params['ffmpeg_settings']
        )
    else:
        pass
    transcode_video_uri = user_data['transcode_video_uri']

    # segmentation
    if user_data['tile_idx'] < config_params['total_tile_num'] and not user_data['generating_background']:
        tile_info, segment_info = tile_segment_info(chunk_info, user_data, user_data['generating_background'])
        segment_video(config_params['ffmpeg_settings'], transcode_video_uri, dst_video_folder, segment_info)
        user_data['tile_idx'] += 1
        user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}
        if user_data['tile_idx'] == config_params['total_tile_num']:
            user_data['generating_background'] = True
            user_data['tile_idx'] = 0

    # resize background stream and segment
    elif user_data['tile_idx'] < config_params['total_tile_num'] and config_params['background_flag'] and user_data['generating_background']:
        # create the background stream
        if user_data['tile_idx'] == 0:
            bg_projection = config_params['background_info']['background_projection_mode']
            if bg_projection == src_projection:
                user_data['bg_video_uri'] = source_video_uri
            else:
                src_resolution = [video_info['height'], video_info['width']]
                bg_resolution = [config_params['background_height'], config_params['background_width']]
                user_data['bg_video_uri'] = transcode_video(
                    source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
                    dst_video_folder, chunk_info, config_params['ffmpeg_settings']
                )

            resize_video(config_params['ffmpeg_settings'], user_data['bg_video_uri'], dst_video_folder, config_params['background_info'])
        tile_info, segment_info = tile_segment_info(chunk_info, user_data, user_data['generating_background'])
        segment_video(config_params['ffmpeg_settings'], user_data['bg_video_uri'], dst_video_folder, segment_info)
        user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}
        user_data['tile_idx'] += 1
    else:
        user_video_spec = None

    return user_video_spec, user_data


def download_decision(network_stats, motion_history, video_size, curr_ts, user_data, video_info):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module

    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    if curr_ts == 0:  # initialize the related parameters
        user_data['next_download_idx'] = 0
        user_data['latest_decision'] = {"high_res": [], "background": []}
        # if user_data['config_params']['background_flag']:
        #     user_data['latest_decision']['background'] = []
    dl_list = []
    chunk_idx = user_data['next_download_idx']
    latest_decision = user_data['latest_decision']

    if user_data['next_download_idx'] >= video_info['duration'] / video_info['chunk_duration']:
        return dl_list, user_data

    predicted_record = predict_motion_tile(motion_history, config_params['motion_history_size'], config_params['motion_prediction_size'])  # motion prediction
    tile_record = tile_decision(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data)     # tile decision
    dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)

    user_data = update_decision_info(user_data, tile_record, curr_ts)            # update decision information

    return dl_list, user_data


def generate_display_result(curr_display_frames, current_display_chunks, curr_fov, dst_video_frame_uri, frame_idx, video_size, user_data, video_info):
    """
    Generate fov images corresponding to different approaches

    Parameters
    ----------
    curr_display_frames: list
        current available video tile frames
    current_display_chunks: list
        current available video chunks
    curr_fov: dict
        current fov information, with format {"curr_motion", "range_fov", "fov_resolution"}
    dst_video_frame_uri: str
        the uri of generated fov frame
    frame_idx: int
        frame index of current display frame
    video_size: dict
        video size of preprocessed video
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for evaluation

    Returns
    -------
    user_data: dict
        updated user_data
    """

    get_logger().debug(f'[evaluation] start get display img {frame_idx}')

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    video_info = user_data['video_info']
    config_params = user_data['config_params']

    chunk_idx = int(frame_idx * (1000 / video_info['video_fps']) // (video_info['chunk_duration'] * 1000))  # frame idx starts from 0
    if chunk_idx <= len(current_display_chunks) - 1:
        tile_list = current_display_chunks[chunk_idx]['tile_list']
    else:
        tile_list = current_display_chunks[-1]['tile_list']

    avail_tile_list = []
    for i in range(len(tile_list)):
        tile_id = tile_list[i]['tile_id']
        tile_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        if type(tile_idx) == str:
            tile_idx = int(tile_idx[:-3]) - config_params['total_tile_num'] # remove _bg and make negative
        avail_tile_list.append(tile_idx)
        get_logger().debug(f'available tile {tile_idx}') # available tiles no problem

    # calculating fov_uv parameters
    fov_ypr = [float(curr_fov['curr_motion']['yaw']), float(curr_fov['curr_motion']['pitch']), 0]
    _3d_polar_coord = fov_to_3d_polar_coord(fov_ypr, curr_fov['range_fov'], curr_fov['fov_resolution'])
    pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [config_params['converted_height'], config_params['converted_width']])

    coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
    get_logger().debug(f'coord tile list: {coord_tile_list.shape}')
    relative_tile_coord = pixel_coord_to_relative_tile_coord(pixel_coord, coord_tile_list, video_size, chunk_idx)
    get_logger().debug(f'relative_tile_coord: {relative_tile_coord}')
    unavail_pixel_coord = ~np.isin(coord_tile_list, avail_tile_list)    # calculate the pixels that have not been transmitted.
    coord_tile_list[unavail_pixel_coord] -= config_params['total_tile_num'] # negative to represent background
    # get_logger().debug(f'coord tile list with bg: {coord_tile_list}')
    # background coords, change with size, so need to change when size adaption
    if config_params['background_flag']:
        background_pixel_coord = _3d_polar_coord_to_pixel_coord(
                    _3d_polar_coord,
                    config_params['background_info']['background_projection_mode'],
                    [config_params['background_height'], config_params['background_width']]
                )
        background_coord_tile_list = pixel_coord_to_tile(background_pixel_coord, config_params['total_tile_num'], video_size, chunk_idx, '_bg')
        get_logger().debug(f'background_coord_tile_list: {background_coord_tile_list.shape}')
        background_relative_tile_coord = pixel_coord_to_relative_tile_coord(background_pixel_coord, background_coord_tile_list, video_size, chunk_idx, '_bg')
        get_logger().debug(f'background_relative_tile_coord: {background_relative_tile_coord}')

    display_img = np.full((coord_tile_list.shape[0], coord_tile_list.shape[1], 3), [128, 128, 128], dtype=np.float32)  # create an empty matrix for the final image

    for i, tile_idx in enumerate(avail_tile_list):
        hit_coord_mask = (coord_tile_list == tile_idx)
        if not np.any(hit_coord_mask):  # if no pixels belong to the current frame, skip
            continue
        get_logger().debug(f'mapping tile {tile_idx}')
        if tile_idx >= 0:
            dstMap_u, dstMap_v = cv2.convertMaps(relative_tile_coord[0].astype(np.float32), relative_tile_coord[1].astype(np.float32), cv2.CV_16SC2)
        else:
            dstMap_u, dstMap_v = cv2.convertMaps(background_relative_tile_coord[0].astype(np.float32), background_relative_tile_coord[1].astype(np.float32), cv2.CV_16SC2)
        remapped_frame = cv2.remap(curr_display_frames[i], dstMap_u, dstMap_v, cv2.INTER_LINEAR)
        display_img[hit_coord_mask] = remapped_frame[hit_coord_mask]
        cv2.imshow(f'{tile_idx}', curr_display_frames[i])

    cv2.imwrite(dst_video_frame_uri, display_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    get_logger().debug(f'[evaluation] end get display img {frame_idx}')

    return user_data


def update_decision_info(user_data, tile_record, curr_ts):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """

    # update latest_decision
    for i in range(len(tile_record['high_res'])):
        if tile_record['high_res'][i] not in user_data['latest_decision']['high_res']:
            user_data['latest_decision']['high_res'].append(tile_record['high_res'][i])
    # if user_data['config_params']['background_flag']:
    #     if -1 not in user_data['latest_decision']:
    #         user_data['latest_decision'].append(-1)
    for i in range(len(tile_record['background'])):
        if tile_record['background'][i] not in user_data['latest_decision']['background']:
            user_data['latest_decision']['background'].append(tile_record['background'][i])

    # update chunk_idx & latest_decision
    if curr_ts == 0 or curr_ts >= user_data['video_info']['pre_download_duration'] \
            + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000:
        user_data['next_download_idx'] += 1
        user_data['latest_decision'] = {"high_res": [], "background": []}

    return user_data


def tile_segment_info(chunk_info, user_data, background):
    """
    Generate the information for the current tile, with required format
    Parameters
    ----------
    chunk_info: dict
        chunk information
    user_data: dict
        user related information
    background: bool
        segmenting background or not

    Returns
    -------
    tile_info: dict
        tile related information, with format {chunk_idx:, tile_idx:}
    segment_info: dict
        segmentation related information, with format
        {segment_out_info:{width:, height:}, start_position:{width:, height:}}
    """

    tile_idx = user_data['tile_idx']

    index_width = tile_idx % user_data['config_params']['tile_width_num']        # determine which col
    index_height = tile_idx // user_data['config_params']['tile_width_num']      # determine which row

    if not background:
        segment_info = {
            'segment_out_info': {
                'width': user_data['config_params']['tile_width'],
                'height': user_data['config_params']['tile_height']
            },
            'start_position': {
                'width': index_width * user_data['config_params']['tile_width'],
                'height': index_height * user_data['config_params']['tile_height']
            }
        }
    else :
        segment_info = {
            'segment_out_info': {
                'width': user_data['config_params']['background_info']['tile_width'],
                'height': user_data['config_params']['background_info']['tile_height']
            },
            'start_position': {
                'width': index_width * user_data['config_params']['background_info']['tile_width'],
                'height': index_height * user_data['config_params']['background_info']['tile_height']
            }
        }

    if not background:
        tile_info = {
            'chunk_idx': user_data['chunk_idx'],
            'tile_idx': tile_idx
        }
    else:
        tile_info = {
            'chunk_idx': user_data['chunk_idx'],
            'tile_idx': f'{str(tile_idx).zfill(3)}_bg'
        }

    return tile_info, segment_info

def tile_decision(predicted_record, video_size, range_fov, chunk_idx, user_data):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: dist
        the decided tile dist of current update, contains high_res and surrounding tiles (for background)
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.
    config_params = user_data['config_params']
    tile_record = {"high_res": [], "background": []}
    sampling_size = [50, 50]
    converted_width = user_data['config_params']['converted_width']
    converted_height = user_data['config_params']['converted_height']
    for predicted_motion in predicted_record:
        _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, sampling_size)
        pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
        unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
        tile_record['high_res'].extend(unique_tile_list)

    # add surrounding tiles of predicted tiles as high-probability tiles
    if config_params['background_flag']:
        # if -1 not in user_data['latest_decision']:
        #     tile_record.append(-1)
        for tile_idx in tile_record['high_res']:
            surrounding_tiles = get_surrounding_tiles(user_data, tile_idx)
            for tile in surrounding_tiles:
                if tile not in tile_record['high_res'] and tile not in tile_record['background']:
                    tile_record['background'].append(tile)

    return tile_record

def generate_dl_list(chunk_idx, tile_record, latest_result, dl_list):
    """
    Based on the decision result, generate the required dl_list to be returned in the specified format.
    (As an example, users can implement their corresponding function.)

    Parameters
    ----------
    chunk_idx: int
        the index of current chunk
    tile_record: dict
        the decided tile dist of current update, contains high_res and surrounding tiles (for background)
    latest_result: dict
        recording the latest decision result, have high_res and background lists
    dl_list: list
        the decided tile list

    Returns
    -------
    dl_list: list
        updated dl_list
    """

    tile_result = {"high_res": [], "background": []}
    # add the predicted high_res tiles
    for i in range(len(tile_record['high_res'])):
        tile_idx = tile_record['high_res'][i]
        if tile_idx not in latest_result['high_res']:
            if tile_idx != -1:
                tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
            else:
                tile_id = f"chunk_{str(chunk_idx).zfill(4)}_background"
            tile_result['high_res'].append(tile_id)
            
    # add the surrounding background tiles
    for bg_tile in tile_record['background']:
        if bg_tile not in latest_result['background'] and bg_tile not in latest_result['high_res']:
            tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(bg_tile).zfill(3)}_bg"
            tile_result['background'].append(tile_id)

    if len(tile_result['high_res']) != 0 or len(tile_result['background']) != 0:
        dl_list.append(
            {
                'chunk_idx': chunk_idx,
                'decision_data': {
                    'tile_info': tile_result['high_res'] + tile_result['background']
                }
            }
        )

    return dl_list


def get_surrounding_tiles(user_data, tile_idx):
    """
    Get the surrounding tiles of a certain tile
    
    Parameter
    ---------
    user_data: dict
        user defined data structure, used for tile_width_num and tile_height_num
    tile_idx: int
        the pivot tile
        
    Returns
    -------
    surrounding_tiles: list
        the list containing surrounding tiles
    """
    tile_width_num = user_data['config_params']['tile_width_num']
    tile_height_num = user_data['config_params']['tile_height_num']
    row = tile_idx // tile_width_num      # determine which row
    col = tile_idx % tile_width_num        # determine which col

    surrounding_tiles = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the current tile
            new_row = (row + dr) % tile_width_num
            new_col = (col + dc) % tile_height_num
            surrounding_tiles.append(new_row * tile_width_num + new_col)
    return surrounding_tiles

def pixel_coord_to_tile(pixel_coord, total_tile_num, video_size, chunk_idx, tile_id_appendix = ''):
    """
    Calculate the corresponding tile, for given pixel coordinates

    Parameters
    ----------
    pixel_coord: array
        pixel coordinates
    total_tile_num: int
        total num of tiles for different approach
    video_size: dict
        video size of preprocessed video
    chunk_idx: int
        chunk index
    tile_id_appendix: str
        append behind name of tile_id

    Returns
    -------
    coord_tile_list: list
        the calculated tile list, for the given pixel coordinates
    """

    coord_tile_list = np.full(pixel_coord[0].shape, 0)
    for i in range(total_tile_num):
        tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(i).zfill(3)}{tile_id_appendix}"
        if tile_id not in video_size:
            continue
        tile_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        if len(tile_id_appendix) > 0:
            tile_idx = tile_idx[:-len(tile_id_appendix)]
        tile_start_width = video_size[tile_id]['user_video_spec']['segment_info']['start_position']['width']
        tile_start_height = video_size[tile_id]['user_video_spec']['segment_info']['start_position']['height']
        tile_width = video_size[tile_id]['user_video_spec']['segment_info']['segment_out_info']['width']
        tile_height = video_size[tile_id]['user_video_spec']['segment_info']['segment_out_info']['height']

        # Create a Boolean mask to check if the coordinates are within the tile range
        mask_width = (tile_start_width <= pixel_coord[0]) & (pixel_coord[0] < tile_start_width + tile_width)
        mask_height = (tile_start_height <= pixel_coord[1]) & (pixel_coord[1] < tile_start_height + tile_height)

        # find coordinates that satisfy both width and height conditions
        hit_coord_mask = mask_width & mask_height

        # update coord_tile_list
        coord_tile_list[hit_coord_mask] = tile_idx

    return coord_tile_list

def pixel_coord_to_relative_tile_coord(pixel_coord, coord_tile_list, video_info, chunk_idx, tile_id_appendix = ''):
    """
    Calculate the relative position of the pixel_coord coordinates on each tile.

    Parameters
    ----------
    pixel_coord: array
        pixel coordinates
    coord_tile_list: list
        calculated tile list
    video_info: dict
    chunk_idx: int
        chunk index
    tile_id_appendix: str
        append behind name of tile_id

    Returns
    -------
    relative_tile_coord: array
        the relative tile coord for the given pixel coordinates
    """

    relative_tile_coord = copy.deepcopy(pixel_coord)
    unique_tile_list = np.unique(coord_tile_list)
    for i in unique_tile_list:
        tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(i).zfill(3)}{tile_id_appendix}"
        tile_start_width = video_info[tile_id]['user_video_spec']['segment_info']['start_position']['width']
        tile_start_height = video_info[tile_id]['user_video_spec']['segment_info']['start_position']['height']
        tile_width = video_info[tile_id]['user_video_spec']['segment_info']['segment_out_info']['width']
        tile_height = video_info[tile_id]['user_video_spec']['segment_info']['segment_out_info']['height']

        hit_coord_mask = (coord_tile_list == i)

        # update the relative position
        relative_tile_coord[0][hit_coord_mask] = np.clip(relative_tile_coord[0][hit_coord_mask] - tile_start_width, 0, tile_width - 1)
        relative_tile_coord[1][hit_coord_mask] = np.clip(relative_tile_coord[1][hit_coord_mask] - tile_start_height, 0, tile_height - 1)

    return relative_tile_coord