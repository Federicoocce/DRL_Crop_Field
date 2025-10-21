import os
import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io
from .config import config

from .utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform

class CARLA_Data(Dataset):

    def __init__(self, root, config, shared_dict=None):

        self.seq_len = np.array(config.seq_len)
        assert (config.img_seq_len == 1)
        self.pred_len = np.array(config.pred_len)

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)
        self.multitask = np.array(config.multitask)
        self.data_cache = shared_dict
        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.use_point_pillars = np.array(config.use_point_pillars)
        self.max_lidar_points = np.array(config.max_lidar_points)
        self.backbone = np.array(config.backbone).astype(np.string_)
        self.inv_augment_prob = np.array(config.inv_augment_prob)
        
        self.converter = np.uint8(config.converter)

        self.images = []
        self.bevs = []
        self.depths = []
        self.semantics = []
        self.lidars = []
        self.labels = []
        self.measurements = []

        for sub_root in tqdm(root, file=sys.stdout):
            sub_root = Path(sub_root)

            # list sub-directories in root
            root_files = os.listdir(sub_root)
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
            for route in routes:
                route_dir = sub_root / route
                num_seq = len(os.listdir(route_dir / "lidar"))

                # ignore the first two and last two frame
                for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                    # load input seq and pred seq jointly
                    image = []
                    bev = []
                    depth = []
                    semantic = []
                    lidar = []
                    label = []
                    measurement= []
                    # Loads the current (and past) frames (if seq_len > 1)
                    for idx in range(self.seq_len):
                        image.append(route_dir / "rgb" / ("%04d.png" % (seq + idx)))
                        bev.append(route_dir / "topdown" / ("encoded_%04d.png" % (seq + idx)))
                        depth.append(route_dir / "depth" / ("%04d.png" % (seq + idx)))
                        semantic.append(route_dir / "semantics" / ("%04d.png" % (seq + idx)))
                        lidar.append(route_dir / "lidar" / ("%04d.npy" % (seq + idx)))
                        measurement.append(route_dir / "measurements" / ("%04d.json"%(seq+idx)))

                    # Additionally load future labels of the waypoints
                    for idx in range(self.seq_len + self.pred_len):
                        label.append(route_dir / "label_raw" / ("%04d.json" % (seq + idx)))

                    self.images.append(image)
                    self.bevs.append(bev)
                    self.depths.append(depth)
                    self.semantics.append(semantic)
                    self.lidars.append(lidar)
                    self.labels.append(label)
                    self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.images       = np.array(self.images      ).astype(np.string_)
        self.bevs         = np.array(self.bevs        ).astype(np.string_)
        self.depths       = np.array(self.depths      ).astype(np.string_)
        self.semantics    = np.array(self.semantics   ).astype(np.string_)
        self.lidars       = np.array(self.lidars      ).astype(np.string_)
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        print("Loading %d lidars from %d folders"%(len(self.lidars), len(root)))

    def __len__(self):
        """Returns the length of the dataset. """
        return self.lidars.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        cv2.setNumThreads(0) # Disable threading because the data loader will already split in threads.

        data = dict()
        backbone = str(self.backbone, encoding='utf-8')

        images = self.images[index]
        bevs = self.bevs[index]
        depths = self.depths[index]
        semantics = self.semantics[index]
        lidars = self.lidars[index]
        labels = self.labels[index]
        measurements = self.measurements[index]

        # load measurements
        loaded_images = []
        loaded_bevs = []
        loaded_depths = []
        loaded_semantics = []
        loaded_lidars = []
        loaded_labels = []
        loaded_measurements = []

        if(backbone == 'geometric_fusion'):
            loaded_lidars_raw = []

        # Because the strings are stored as numpy byte objects we need to convert them back to utf-8 strings
        # Since we also load labels for future timesteps, we load and store them separately
        for i in range(self.seq_len+self.pred_len):
            if ((not (self.data_cache is None)) and (str(labels[i], encoding='utf-8') in self.data_cache)):
                    labels_i = self.data_cache[str(labels[i], encoding='utf-8')]
            else:

                with open(str(labels[i], encoding='utf-8'), 'r') as f2:
                    labels_i = ujson.load(f2)

                if not self.data_cache is None:
                    self.data_cache[str(labels[i], encoding='utf-8')] = labels_i

            loaded_labels.append(labels_i)


        for i in range(self.seq_len):
            if not self.data_cache is None and str(measurements[i], encoding='utf-8') in self.data_cache:
                    measurements_i, images_i, lidars_i, lidars_raw_i, bevs_i, depths_i, semantics_i = self.data_cache[str(measurements[i], encoding='utf-8')]
                    images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
                    depths_i = cv2.imdecode(depths_i, cv2.IMREAD_UNCHANGED)
                    semantics_i = cv2.imdecode(semantics_i, cv2.IMREAD_UNCHANGED)
                    bevs_i.seek(0) # Set the point to the start of the file like object
                    bevs_i = np.load(bevs_i)['arr_0']
            else:
                with open(str(measurements[i], encoding='utf-8'), 'r') as f1:
                    measurements_i = ujson.load(f1)

                lidars_i = np.load(str(lidars[i], encoding='utf-8'), allow_pickle=True)[1]  # [...,:3] # lidar: XYZI
                if (backbone == 'geometric_fusion'):
                    lidars_raw_i = np.load(str(lidars[i], encoding='utf-8'), allow_pickle=True)[1][..., :3]  # lidar: XYZI
                else:
                    lidars_raw_i = None
                lidars_i[:, 1] *= -1

                images_i = cv2.imread(str(images[i], encoding='utf-8'), cv2.IMREAD_COLOR)
                if(images_i is None):
                    print("Error loading file: ", str(images[i], encoding='utf-8'))
                images_i = scale_image_cv2(cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB), self.scale)

                bev_array = cv2.imread(str(bevs[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                bev_array = cv2.cvtColor(bev_array, cv2.COLOR_BGR2RGB)
                if (bev_array is None):
                    print("Error loading file: ", str(bevs[i], encoding='utf-8'))
                bev_array = np.moveaxis(bev_array, -1, 0)
                bevs_i = decode_pil_to_npy(bev_array).astype(np.uint8)
                if self.multitask:
                    depths_i = cv2.imread(str(depths[i], encoding='utf-8'), cv2.IMREAD_COLOR)
                    if (depths_i is None):
                        print("Error loading file: ", str(depths[i], encoding='utf-8'))
                    depths_i = scale_image_cv2(cv2.cvtColor(depths_i, cv2.COLOR_BGR2RGB), self.scale)

                    semantics_i = cv2.imread(str(semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                    if (semantics_i is None):
                        print("Error loading file: ", str(semantics[i], encoding='utf-8'))
                    semantics_i = scale_seg(semantics_i, self.scale)
                else:
                    depths_i = None
                    semantics_i = None

                if not self.data_cache is None:
                    # We want to cache the images in png format instead of uncompressed, to reduce memory usage
                    result, compressed_imgage = cv2.imencode('.png', images_i)
                    result, compressed_depths = cv2.imencode('.png', depths_i)
                    result, compressed_semantics = cv2.imencode('.png', semantics_i)
                    compressed_bevs = io.BytesIO()  # bev has 2 channels which does not work with png compression so we use generic numpy in memory compression
                    np.savez_compressed(compressed_bevs, bevs_i)
                    self.data_cache[str(measurements[i], encoding='utf-8')] = (measurements_i, compressed_imgage, lidars_i, lidars_raw_i, compressed_bevs, compressed_depths, compressed_semantics)

            loaded_images.append(images_i)
            loaded_bevs.append(bevs_i)
            loaded_depths.append(depths_i)
            loaded_semantics.append(semantics_i)
            loaded_lidars.append(lidars_i)
            loaded_measurements.append(measurements_i)
            if (backbone == 'geometric_fusion'):
                loaded_lidars_raw.append(lidars_raw_i)

        labels = loaded_labels
        measurements = loaded_measurements

        # load image, only use current frame
        # augment here
        crop_shift = 0
        degree = 0
        rad = np.deg2rad(degree)
        do_augment = self.augment and random.random() > self.inv_augment_prob
        if do_augment:
            degree = (random.random() * 2. - 1.) * self.aug_max_rotation
            rad = np.deg2rad(degree)
            crop_shift = degree / 60 * self.img_width / self.scale # we scale first

        images_i = loaded_images[self.seq_len-1]
        images_i = crop_image_cv2(images_i, crop=self.img_resolution, crop_shift=crop_shift)

        bevs_i = load_crop_bev_npy(loaded_bevs[self.seq_len-1], degree)
        
        data['rgb'] = images_i
        data['bev'] = bevs_i

        if self.multitask:
            depths_i = loaded_depths[self.seq_len-1]
            depths_i = get_depth(crop_image_cv2(depths_i, crop=self.img_resolution, crop_shift=crop_shift))

            semantics_i = loaded_semantics[self.seq_len-1]
            semantics_i = self.converter[crop_seg(semantics_i, crop=self.img_resolution, crop_shift=crop_shift)]

            data['depth'] = depths_i
            data['semantic'] = semantics_i

        # need to concatenate seq data here and align to the same coordinate
        lidars = []
        if (backbone == 'geometric_fusion'):
            lidars_raw = []
        if (self.use_point_pillars == True):
            lidars_pillar = []

        for i in range(self.seq_len):
            lidar = loaded_lidars[i]
            # transform lidar to lidar seq-1
            lidar = align(lidar, measurements[i], measurements[self.seq_len-1], degree=degree)
            lidar_bev = lidar_to_histogram_features(lidar)
            lidars.append(lidar_bev)

            if (backbone == 'geometric_fusion'):
                # We don't align the raw LiDARs for now
                lidar_raw = loaded_lidars_raw[i]
                lidars_raw.append(lidar_raw)

            if (self.use_point_pillars == True):
                # We want to align the LiDAR for the point pillars, but not voxelize them
                lidar_pillar = deepcopy(loaded_lidars[i])
                lidar_pillar = align(lidar_pillar, measurements[i], measurements[self.seq_len-1], degree=degree)
                lidars_pillar.append(lidar_pillar)

        # NOTE: This flips the ordering of the LiDARs since we only use 1 it does nothing. Can potentially be removed.
        lidar_bev = np.concatenate(lidars[::-1], axis=0)
        if (backbone == 'geometric_fusion'):
            lidars_raw = np.concatenate(lidars_raw[::-1], axis=0)
        if (self.use_point_pillars == True):
            lidars_pillar = np.concatenate(lidars_pillar[::-1], axis=0)

        if (backbone == 'geometric_fusion'):
            curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(lidars_raw), debug=False)


        # ego car is always the first one in label file
        ego_id = labels[self.seq_len-1][0]['id']

        # only use label of frame 1
        bboxes = parse_labels(labels[self.seq_len-1], rad=-rad)
        waypoints = get_waypoints(labels[self.seq_len-1:], self.pred_len+1)
        waypoints = transform_waypoints(waypoints)

        # save waypoints in meters
        filtered_waypoints = []
        for id in list(bboxes.keys()) + [ego_id]:
            waypoint = []
            for matrix, flag in waypoints[id][1:]:
                waypoint.append(matrix[:2, 3])
            filtered_waypoints.append(waypoint)
        waypoints = np.array(filtered_waypoints)

        label = []
        for id in bboxes.keys():
            label.append(bboxes[id])
        label = np.array(label)
        
        # padding
        label_pad = np.zeros((20, 7), dtype=np.float32)
        ego_waypoint = waypoints[-1]

        # for the augmentation we only need to transform the waypoints for ego car
        degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                              [-np.sin(rad), np.cos(rad)]])
        ego_waypoint = (degree_matrix @ ego_waypoint.T).T

        if label.shape[0] > 0:
            label_pad[:label.shape[0], :] = label

        if(self.use_point_pillars == True):
            # We need to have a fixed number of LiDAR points for the batching to work, so we pad them and save to total amound of real LiDAR points.
            fixed_lidar_raw = np.empty((self.max_lidar_points, 4), dtype=np.float32)
            num_points = min(self.max_lidar_points, lidars_pillar.shape[0])
            fixed_lidar_raw[:num_points, :4] = lidars_pillar
            data['lidar_raw'] = fixed_lidar_raw
            data['num_points'] = num_points

        if (backbone == 'geometric_fusion'):
            data['bev_points'] = curr_bev_points
            data['cam_points'] = curr_cam_points

        data['lidar'] = lidar_bev
        data['label'] = label_pad
        data['ego_waypoint'] = ego_waypoint

        # other measurement
        # do you use the last frame that already happend or use the next frame?
        data['steer'] = measurements[self.seq_len-1]['steer']
        data['throttle'] = measurements[self.seq_len-1]['throttle']
        data['brake'] = measurements[self.seq_len-1]['brake']
        data['light'] = measurements[self.seq_len-1]['light_hazard']
        data['speed'] = measurements[self.seq_len-1]['speed']
        data['theta'] = measurements[self.seq_len-1]['theta']
        data['x_command'] = measurements[self.seq_len-1]['x_command']
        data['y_command'] = measurements[self.seq_len-1]['y_command']

        # target points
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        ego_theta = measurements[self.seq_len-1]['theta'] + rad # + rad for augmentation
        ego_x = measurements[self.seq_len-1]['x']
        ego_y = measurements[self.seq_len-1]['y']
        x_command = measurements[self.seq_len-1]['x_command']
        y_command = measurements[self.seq_len-1]['y_command']
        
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        local_command_point = R.T.dot(local_command_point)

        data['target_point'] = local_command_point
        
        data['target_point_image'] = draw_target_point(local_command_point)
        return data

def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized
    #clip to 50 meters
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 # Rescale map to lie in [0,1]

    return normalized


def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

# this is only for visualization, For training, we should use vehicle coordinate

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    T = get_vehicle_to_virtual_lidar_transform()
    
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
            
    return waypoints

def align(lidar_0, degree=0):
    """
    Applies a 2D rotation to the LiDAR point cloud for data augmentation.
    This version is simplified for ROS and removes all CARLA-specific transforms.

    Args:
        lidar_0 (np.array): The input LiDAR point cloud (Nx3 or Nx4).
        degree (float): The rotation angle in degrees.

    Returns:
        np.array: The rotated LiDAR point cloud.
    """
    # Create a 2D rotation matrix from the augmentation angle.
    # In ROS (X-fwd, Y-left), a positive rotation is counter-clockwise.
    rad = np.deg2rad(degree)
    degree_matrix_2d = np.array([[np.cos(rad), -np.sin(rad)],
                                 [np.sin(rad),  np.cos(rad)]])

    # Apply the rotation to the X and Y coordinates of the point cloud.
    # The Z coordinate (lidar_0[:, 2]) remains unchanged.
    lidar_rotated_xy = (degree_matrix_2d @ lidar_0[:, :2].T).T
    
    # Re-assemble the point cloud with the rotated XY and original Z/Intensity
    rotated_lidar = np.hstack((lidar_rotated_xy, lidar_0[:, 2:]))
    
    return rotated_lidar

def lidar_to_histogram_features(point_cloud_np, crop=256):
    """
    Converts a (potentially rotated) LiDAR point cloud to a BEV histogram.
    """
    if point_cloud_np.shape[0] == 0:
         return np.zeros((crop, crop, 2), dtype=np.uint8)

    # --- Parameters (Same as before) ---
    x_max_meters = 4.0  # Forward range
    y_max_meters = 2.0  # Side range (+/-)
    pixels_per_meter = 16
    hist_max_per_pixel = 5
    z_threshold = 0.0

    # Define boundaries
    x_bins = np.linspace(0, x_max_meters, int(x_max_meters * pixels_per_meter) + 1)
    y_bins = np.linspace(-y_max_meters, y_max_meters, int(y_max_meters * 2 * pixels_per_meter) + 1)

    # Split & Histogram
    below = point_cloud_np[point_cloud_np[..., 2] <= z_threshold]
    above = point_cloud_np[point_cloud_np[..., 2] > z_threshold]

    below_hist = np.histogram2d(below[..., 1], below[..., 0], bins=(y_bins, x_bins))[0]
    above_hist = np.histogram2d(above[..., 1], above[..., 0], bins=(y_bins, x_bins))[0]

    # Normalize & Clip
    below_hist[below_hist > hist_max_per_pixel] = hist_max_per_pixel
    above_hist[above_hist > hist_max_per_pixel] = hist_max_per_pixel
    
    below_features = (below_hist / hist_max_per_pixel * 255).astype(np.uint8)
    above_features = (above_hist / hist_max_per_pixel * 255).astype(np.uint8)

    # Orient correctly (Y-forward as rows, 0 at bottom)
    features = np.stack([np.flipud(below_features.T), np.flipud(above_features.T)], axis=-1)

    # Ensure exact output size
    h, w, c = features.shape
    output = np.zeros((crop, crop, 2), dtype=np.uint8)
    # Center result if bins resulted in smaller image
    start_h = (crop - h) // 2
    start_w = (crop - w) // 2
    output[start_h:start_h+h, start_w:start_w+w, :] = features
    
    return output

def draw_target_point(target_point_world, crop=256):
    """
    Draws the target point on a BEV canvas that has the EXACT same dimensions
    and scale as the LiDAR BEV histogram.
    This ensures that the target point image can be correctly overlaid on the LiDAR BEV.
    """
    # These parameters MUST be identical to those in lidar_to_histogram_features
    x_max_meters = 4.0
    y_max_meters = 2.0  # Side range (+/-)
    pixels_per_meter = 16

    # Create the base canvas, matching the final output size of the LiDAR BEV
    image = np.zeros((crop, crop), dtype=np.uint8)

    # Unpack the world coordinates (ROS standard frame: X is forward, Y is left)
    robot_x_forward, robot_y_left = target_point_world
    
    # --- Coordinate to Pixel Mapping ---
    # This logic mirrors the binning and orientation of lidar_to_histogram_features.
    
    # 1. Calculate the feature map size (before padding to `crop`)
    feature_map_height = int(x_max_meters * pixels_per_meter)      # Bins along the X (forward) axis
    feature_map_width = int(y_max_meters * 2 * pixels_per_meter)  # Bins along the Y (left/right) axis

    # 2. Map robot's Y-coordinate (left/right) to a pixel column in the feature map
    # The range is [-y_max_meters, y_max_meters].
    # We map this to [0, feature_map_width].
    pixel_col_feature = int((-robot_y_left + y_max_meters) * pixels_per_meter)
    
    # 3. Map robot's X-coordinate (forward) to a pixel row in the feature map
    # The range is [0, x_max_meters].
    # We map this to [0, feature_map_height] and then flip it because the origin is at the bottom.
    pixel_row_feature = int(feature_map_height - 1 - (robot_x_forward * pixels_per_meter))
    
    # 4. Calculate the final padded pixel coordinates on the `crop x crop` canvas
    # The feature map is centered within the larger canvas.
    start_h = (crop - feature_map_height) // 2
    start_w = (crop - feature_map_width) // 2
    
    final_pixel_row = start_h + pixel_row_feature
    final_pixel_col = start_w + pixel_col_feature
    
    # Create the point tuple (column, row) for OpenCV
    point_pixel = (final_pixel_col, final_pixel_row)

    # Draw the circle, ensuring it's within the image bounds
    if 0 <= final_pixel_row < crop and 0 <= final_pixel_col < crop:
        cv2.circle(image, point_pixel, radius=5, color=255, thickness=-1)
    
    # Reshape for PyTorch (C, H, W)
    image = image.reshape(1, crop, crop)
    return image.astype(np.float32) / 255.0
def get_bbox_label(bbox, rad=0):
    # dx, dy, dz, x, y, z, yaw
    # ignore z
    dz, dx, dy, x, y, z, yaw, speed, brake =  bbox

    pixels_per_meter = 8

    # augmentation
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                              [-np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position

    position = np.clip(position, 0., 255.)
    x, y = position[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, dy*pixels_per_meter, dx*pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox


def parse_labels(labels, rad=0):
    bboxes = {}
    for result in labels:
        num_points = result['num_points']
        distance = result['distance']

        x = result['position'][0]
        y = result['position'][1]

        bbox = result['extent'] + result['position'] + [result['yaw'], result['speed'], result['brake']]
        bbox = get_bbox_label(bbox, rad)

        # Filter bb that are outside of the LiDAR after the random augmentation. The bounding box is now in image space
        if num_points <= 1 or bbox[0] <= 0.0 or bbox[0] >= 255.0 or bbox[1] <= 0.0 or bbox[1] >=255.0:
            continue

        bboxes[result['id']] = bbox
    return bboxes

def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized

def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized

def crop_image(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    
    # only shift for x direction
    start_x += int(crop_shift)

    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def scale_seg(image, scale):
    (width, height) = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    if scale != 1:
        im_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        im_resized = image
    return im_resized

def crop_seg(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a seg image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop

    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cropped_image

def load_crop_bev_npy(bev_array, degree):
    """
    Load and crop an Image.
    Crop depends on augmentation angle.
    """
    PIXELS_PER_METER_FOR_BEV = 5
    PIXLES = 32 * PIXELS_PER_METER_FOR_BEV
    start_x = 250 - PIXLES // 2
    start_y = 250 - PIXLES

    # shift the center by 7 because the lidar is + 1.3 in x 
    bev_array = np.moveaxis(bev_array, 0, -1).astype(np.float32)
    bev_shift = np.zeros_like(bev_array)
    bev_shift[7:] = bev_array[:-7]

    bev_shift = rotate(bev_shift, degree)
    cropped_image = bev_shift[start_y:start_y+PIXLES, start_x:start_x+PIXLES]
    cropped_image = np.moveaxis(cropped_image, -1, 0)

    # we need to predict others so append 0 to the first channel
    cropped_image = np.concatenate((np.zeros_like(cropped_image[:1]), 
                                    cropped_image[:1],
                                    cropped_image[:1] + cropped_image[1:2]), axis=0)

    cropped_image = np.argmax(cropped_image, axis=0)
    
    return cropped_image





def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    """
    Compute projections between LiDAR BEV and image space
    """
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))

    tmp_bev = np.empty((lidar_x, lidar_y, ), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y, ), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i,j] = []

    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []

    for i in range(valid_bev_points.shape[0]):
        tmp_bev[valid_bev_points[i][0]//scale, valid_bev_points[i][1]//scale].append(valid_cam_points[i]//scale)
        tmp_cam[valid_cam_points[i][0]//scale, valid_cam_points[i][1]//scale].append(valid_bev_points[i]//scale)

    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i,j]

            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i,j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i,j,:num_points] = np.array(cam_to_bev_points)

    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i,j]

            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i,j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i,j,:num_points] = np.array(bev_to_cam_points)

    return cam_to_bev_proj_locs, bev_to_cam_proj_locs

def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from CARLA in the CARLA coordinate system: x left, y forward, z up (LiDAR rotated by 90 degree)
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """

    pixels_per_meter = 8
    lidar_width      = 256
    lidar_height     = 256
    lidar_meters_x   = (lidar_width  / pixels_per_meter) / 2 # Divided by two because the LiDAR is in the center of the image
    lidar_meters_y   =  lidar_height / pixels_per_meter

    downscale_factor = 32

    img_width  = 352
    img_height = 160
    fov_width  = 60

    left_camera_rotation  = -60.0
    right_camera_rotation =  60.0

    fov_height = 2.0 * np.arctan((img_height / img_width) * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)

    # Our pixels are squares so focal_x = focal_y
    focal_x = img_width  / (2.0 * np.tan(np.deg2rad(fov_width)  / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))

    cam_z   = 2.3
    lidar_z = 2.5

    # get valid points in 64x64 grid
    world[:, 0] *= -1  # flip x axis, so that the positive direction points towards right. new coordinate system: x right, y forward, z up
    lidar = world[abs(world[:,0])<lidar_meters_x] # 32m to the sides
    lidar = lidar[lidar[:,1]<lidar_meters_y] # 64m to the front
    lidar = lidar[lidar[:,1]>0] # 0m to the back

    # Translate Lidar cloud to the same coordinate system as the cameras (They only differ in height)
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)

    # Make copies because we will rotate the new pointclouds
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)


    lidar_indices = np.arange(0, lidar.shape[0], 1)
    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar[..., 1]
    x = ((focal_x * lidar[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar[..., 2]) / z) + (img_height / 2.0)
    result_center = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
    result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]

    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)

    # Rotate the left camera to align with the axis for projection with a pinhole camera model
    theta = np.radians(left_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_left_camera[..., 1]
    x = ((focal_x * lidar_for_left_camera[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar_for_left_camera[..., 2]) / z) + (img_height / 2.0)
    result_left = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
    result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
    result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)

    # Do the same for the right image
    theta = np.radians(right_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_right_camera[..., 1]
    x = ((focal_x * lidar_for_right_camera[..., 0]) / z) + (img_width / 2.0)
    y = ((focal_y * lidar_for_right_camera[..., 2]) / z) + (img_height / 2.0)
    result_right = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] # Cut of right part, it's not used.
    result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width

    # Combine the three images into one
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    if(debug == True):
        # Visualize LiDAR hits in image
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        vis_original_image = image_vis[0].detach().cpu().numpy()
        vis_original_image = np.transpose(vis_original_image, (1, 2, 0)) / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        lidar_vis = lidar_vis.detach().cpu().numpy()
        vis_original_lidar[np.greater(lidar_vis[0,0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[0,1], 0)] = 255


    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        # Project the LiDAR point to BEV and save index of the BEV image pixel.
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height-1)) * -1

        valid_bev_points.append([bev_x, bev_y])
        # Calculate index in the final image by rounding down
        img_x = int(results_total[i][0])
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])


        if (debug == True):
            vis_original_image[img_y, img_x] = np.array([0.0,1.0,0.0])
            vis_bev[bev_y, bev_x] = 255 #Debug visualization
            vis[img_y, img_x] = 255

    if (debug == True):
        # NOTE add the paths you want the images to land in here before debugging
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()


    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)

    return bev_points, cam_points

def decode_pil_to_npy(img):
    """
    """
    (channels, width, height) = (15, img.shape[1], img.shape[2])

    bev_array = np.zeros([channels, width, height])

    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos

    # hard coded to select
    return bev_array[10:12]
def render_sensor_data(camera_image, lidar_bev, camera_image_rear=None):
    """
    Displays sensor data in pop-up windows.
    This version is updated to show the front camera, an optional rear camera,
    and the LiDAR BEV.
    
    Args:
        camera_image (np.array): The front camera image (H, W, 3).
        lidar_bev (np.array): The processed LiDAR BEV (H, W, 2).
        camera_image_rear (np.array, optional): The rear camera image. Defaults to None.
    """
    try:
        # --- Front Camera Visualization ---
        if camera_image is not None:
            img_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            img_display = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Front Camera View", img_display) # Renamed for clarity

        # --- NEW: Rear Camera Visualization ---
        if camera_image_rear is not None:
            img_rear_bgr = cv2.cvtColor(camera_image_rear, cv2.COLOR_RGB2BGR)
            img_rear_display = cv2.resize(img_rear_bgr, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Rear Camera View", img_rear_display)

        # --- BEV Visualization ---
        if lidar_bev is not None:
            # Ensure it's the 2-channel processed BEV
            if lidar_bev.ndim == 3 and lidar_bev.shape[2] == 2:
                bev_h, bev_w, _ = lidar_bev.shape
                bev_display = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
                
                # Channel 0 (below points) -> Mapped to Blue channel
                bev_display[:, :, 0] = lidar_bev[:, :, 0]
                # Channel 1 (above points) -> Mapped to Red channel
                bev_display[:, :, 2] = lidar_bev[:, :, 1]
                
                bev_display_resized = cv2.resize(bev_display, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("LiDAR BEV", bev_display_resized)

        # Update all OpenCV windows if any of them have been created
        if camera_image is not None or lidar_bev is not None or camera_image_rear is not None:
            cv2.waitKey(1)

    except Exception as e:
        # Using print here as this is a utility file without a ROS logger
        print(f"Error in render_sensor_data: {e}")

def close_windows():
    cv2.destroyAllWindows()