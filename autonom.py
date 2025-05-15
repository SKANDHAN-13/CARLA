#!/usr/bin/env python

"""
CARLA Autonomous Driving with LiDAR-based Obstacle Detection
"""



import glob
import numpy as np
import weakref
import os
import sys
import time
import cv2
from datetime import datetime
import argparse
import pygame
import math
from pygame.locals import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name
##############################################################################################

class SemanticCamera:
    def __init__(self, parent_actor, output_dir, hud=None):
        self.parent = parent_actor
        self.output_dir = output_dir  # Can be used in the future for saving images and hence, debugging
        self.hud = hud
        self.rgb_image = None
        self.lidar_data = None
        self.lane_pixel_boundaries = []
        self.lane_boundaries = []
        self.drivable_area = []
        self.obstacles = []
        self._setup_sensors()

    def _setup_sensors(self):
        world = self.parent.get_world()
        bp_library = world.get_blueprint_library()
        
        # RGB Camera
        bp_rgb = bp_library.find('sensor.camera.rgb')
        bp_rgb.set_attribute('image_size_x', '800')
        bp_rgb.set_attribute('image_size_y', '600')
        bp_rgb.set_attribute('fov', '90')
        
        # LiDAR Sensor
        bp_lidar = bp_library.find('sensor.lidar.ray_cast')
        bp_lidar.set_attribute('channels', '64')
        bp_lidar.set_attribute('points_per_second', '500000')
        bp_lidar.set_attribute('rotation_frequency', '20')
        bp_lidar.set_attribute('range', '50')

        # Calculate focal length from FOV
        self.focal_length = 800 / (2 * math.tan(math.radians(90/2)))

        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.rgb_sensor = world.spawn_actor(bp_rgb, camera_transform, attach_to=self.parent)
        
        lidar_transform = carla.Transform(carla.Location(x=1.6, z=2.0))
        self.lidar_sensor = world.spawn_actor(bp_lidar, lidar_transform, attach_to=self.parent)

        weak_self = weakref.ref(self)     
        self.rgb_sensor.listen(lambda image: self._process_rgb_image(image))
        self.lidar_sensor.listen(lambda data: self._process_lidar(data))
        self._recording = False
        self.frame_count = 0
    
    """ # Debugging method used for recording images
    def toggle_recording(self):
        self._recording = not self._recording
        return self._recording
    """

    def detect_lane_lines(self, image):
        """Detect lane boundaries using Line Segment Detector (LSD) algorithm
    with adaptive thresholding and region-of-interest masking
    

    Threshold values determined through empirical testing in 
    urban CARLA environments with varying lighting conditions   
    
    """
        #print("Running lane detection")
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([255, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_yellow = np.array([15, 50, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_img = cv2.bitwise_and(image, image, mask=combined_mask)
        gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)

        """ Applies adaptive thresholding to create a binary image and defines a region of interest (ROI) for lane detection."""
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size - adjust as needed
            2    # Constant subtracted from mean - adjust as needed
        )
        
        
        center_width_percentage = 1
        side_margin = int((1 - center_width_percentage) * width / 2)
        
        roi_vertices = np.array([
            [(side_margin, height), 
            (side_margin, height//2 + 30), 
            (width - side_margin, height//2 + 30), 
            (width - side_margin, height)]
        ], dtype=np.int32)
        
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_binary = cv2.bitwise_and(binary, mask)
        
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(masked_binary)[0]
        
        c=0
        lane_lines=[]
        if lines is not None:
            for line in lines:
                
                x1, y1, x2, y2 = map(int, line[0])
                #length = math.sqrt((x2-x1)**2+(y2-y1)**2)
                #print(length)
                #cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if x2!=x1:
                    c+=1
                    #cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    t = math.atan(((y2-y1)/(x2-x1)))
                    #print(t*math.pi/180,c)
                    length = math.sqrt((x2-x1)**2+(y2-y1)**2)
                    # Parameter tuning - Slope and Length of line values (Depending on fov, focal length and position of the camera)
                    if (math.fabs(t)*math.pi/180>0.02) or (math.fabs(t)<0.7 and math.fabs(t)>0.3) and length>50 :
                        lane_lines.append([t,line[0]])
                        #cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        left,right = [], []
        for k in lane_lines:
                if k[0]<0:
                    left.append(k[1])
                else:
                    right.append(k[1])
        
        #Selecting relevant lines
        rlanes,llanes = [],[] 
        rightmost_left_line, leftmost_right_line = [],[]               
        for le in left:
                x1, y1, x2, y2 = map(int, le)
                if math.sqrt((x1-x2)**2+(y1-y2)**2)>180:
                    llanes.append(le)                            #cv2.line(line_image, (x1, y1), (180+x1, y1), (255, 255, 0), 2)
        for ri in right:
                x1, y1, x2, y2 = map(int, ri)
                if math.sqrt((x1-x2)**2+(y1-y2)**2)>180:
                    rlanes.append(ri)                            #cv2.line(line_image, (x1, y1), (-180+x1, y1), (0, 255, 255), 2)  
        if llanes != [] and rlanes != []:
            rightmost_left_line = max(llanes, key=lambda line: line[0])
            leftmost_right_line = min(rlanes, key=lambda line: line[0])
        elif rlanes!=[] and llanes == []:
            
            leftmost_right_line = min(rlanes, key=lambda line: line[0])
            x1,y1, x2, y2 = leftmost_right_line
            d={}
            d[x1] = y1
            d[x2] = y2
            xo = max(x1,x2)
            xi = min(x1,x2)
            rightmost_left_line = [xo-640,d[xo], xi-200, d[xi]]  
            
        elif llanes!=[] and rlanes == []:
            rightmost_left_line = max(llanes, key=lambda line: line[0])
            x1,y1, x2, y2 = rightmost_left_line
            d={}
            d[x1] = y1
            d[x2] = y2
            xo = max(x1,x2)
            xi = min(x1,x2)
            leftmost_right_line=[xo+200,d[xo], xi+640, d[xi]]

    
        self.lane_boundaries = [rightmost_left_line, leftmost_right_line]
        for i, lane in enumerate(self.lane_boundaries):
            if len(lane) == 4:  # x1,y1,x2,y2 in pixels
                # Camera parameters
                f = 800
                h = 1.7  # Camera height from ground
                width = image.shape[1]
                
                # Convert pixel coordinates to vehicle-local meters
                def pixel_to_meter(u, v):
                    x = (u - width/2) * h / f  # Lateral position
                    y = h * f / (v - image.shape[0]/2) if (v - image.shape[0]/2) != 0 else 20.0  # Forward distance
                    return (x, y)
                
                x1, y1 = pixel_to_meter(lane[0], lane[1])
                x2, y2 = pixel_to_meter(lane[2], lane[3])
                
                # Update lane boundaries with local coordinates
                self.lane_boundaries[i] = [x1, y1, x2, y2]
        self.lane_pixel_boundaries = [rightmost_left_line, leftmost_right_line]
        

    def set_drivable_lane_centers(self, lane_centers):
        """Drivable area from current lane centers"""
        vehicle_loc = self.parent.get_location()
        lookahead_distance = 8.0
        rear_offset = 1.5  

        self.drivable_area = []

        for center in lane_centers:
            # Back and front points along the lane center
            rear = carla.Location(x=center, y=vehicle_loc.y - rear_offset, z=vehicle_loc.z)
            front = carla.Location(x=center, y=vehicle_loc.y + lookahead_distance, z=vehicle_loc.z)
            self.drivable_area.append(rear)
        
        # Close polygon by reversing
        for center in reversed(lane_centers):
            rear = carla.Location(x=center, y=vehicle_loc.y - rear_offset, z=vehicle_loc.z)
            front = carla.Location(x=center, y=vehicle_loc.y + lookahead_distance, z=vehicle_loc.z)
            self.drivable_area.append(front)
    
    def _process_lidar(self, point_cloud):
        """Processing LiDAR using DBSCAN clustering with camera coordinate transformation"""
        self.lidar_timestamp = point_cloud.timestamp
        self.lidar_vehicle_transform = self.parent.get_transform()
        
        # Get LiDAR to camera transform
        lidar_to_world = self.lidar_sensor.get_transform()
        camera_to_world = self.rgb_sensor.get_transform()
        world_to_camera = camera_to_world.get_inverse_matrix()
        
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(data, (-1, 4))[:, :3]
        
        if len(points) == 0:
            self.obstacles = []
            return

        # Transform LiDAR points to camera coordinates
        camera_points = []
        vehicle_points = []
        for point in points:
            # Skip ground and very close points
            if 2>point[2] > -2.05 and np.linalg.norm(point[:2]) > 1.0:
                # Transform to vehicle coordinates (x=right, y=forward, z=up)
                vehicle_point = np.dot(self.lidar_sensor.get_transform().get_matrix(), 
                                    [point[0], point[1], point[2], 1])[:3]
                vehicle_points.append(vehicle_point)
        
        if not vehicle_points:
            self.obstacles = []
            return
            
        vehicle_points = np.array(vehicle_points)
        
        # Filter to only keep points in front-right semi-circle
        front_left_points = []
        rear_points = []
        
        for point in vehicle_points:
            angle = math.degrees(math.atan2(point[1], point[0]))  # x=right, y=forward
            
            # Front-left sector (15-45 degrees left)
            if 0 <= angle <= 80 and point[1] > 0:
                front_left_points.append(point)
                
            # Rear sector (135-225 degrees)
            if angle >= -40  and point[1] < 0:
                rear_points.append(point)
        
        # Combine with existing front-right points
        all_points = front_left_points  + rear_points
        
        if not all_points:
            self.obstacles = []
            return
            
        camera_points = np.array(all_points)
        
        # DBSCAN clustering in camera space
        eps = 0.3
        min_samples = 5
        clusters = []
        visited = set()
        noise = set()

        for i in range(len(camera_points)):
            if i in visited:
                continue
                
            visited.add(i)
            neighbours = self._range_query(camera_points, i, eps)
            
            if len(neighbours) < min_samples:
                noise.add(i)
            else:
                cluster = []
                self._expand_cluster(camera_points, i, neighbours, cluster, visited, eps, min_samples)
                clusters.append(cluster)

        # Extract obstacle properties in camera space
        self.obstacles = []
        for cluster in clusters:
            cluster_pts = camera_points[cluster]
            x_min = np.min(cluster_pts[:, 0])
            x_max = np.max(cluster_pts[:, 0])
            centroid = np.mean(cluster_pts, axis=0)
            
            self.obstacles.append({
                'center': centroid,
                'span': x_max - x_min
            })
        
        self.obstacles = self._merge_close_obstacles(self.obstacles, max_dist=2.0)
        self.obstacles = [obs for obs in self.obstacles if abs(obs['center'][0]) <= 0.5]
        
    def _range_query(self, points, idx, eps):
        """Find all points within eps distance of target point"""
        neighbours = []
        target = points[idx]
        for i, point in enumerate(points):
            if np.linalg.norm(point[:2] - target[:2]) <= eps:
                neighbours.append(i)
        return neighbours

    def _expand_cluster(self, points, idx, neighbors, cluster, visited, eps, min_samples):
        """Recursively expand cluster"""
        cluster.append(idx)
        
        for i in neighbors:
            if i not in visited:
                visited.add(i)
                new_neighbors = self._range_query(points, i, eps)
                
                if len(new_neighbors) >= min_samples:
                    neighbors += [n for n in new_neighbors if n not in neighbors]
                    
            if i not in cluster:
                cluster.append(i)


    def _cluster_obstacles(self, points):
        """Cluster LiDAR points into obstacles"""
        obstacles = []
        if len(points) == 0:
            return obstacles
        
       
        grid_size = 0.5
        grid = {}
        for point in points:
            cell = (int(point[0]/grid_size), int(point[1]/grid_size))
            grid.setdefault(cell, []).append(point)
        
        for cell, points in grid.items():
            if len(points) >40: 
                points = np.array(points)
                centroid = np.mean(points, axis=0)
                x_length = np.max(points[:,0]) - np.min(points[:,0])
                obstacles.append({
                    'center': centroid,
                    'span': x_length
                })
        
        return obstacles
    
    def _merge_close_obstacles(self, obstacles, max_dist=2.0):
        merged = []
        used = set()
        
        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
                
            cluster = [obs1]
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                distance = np.linalg.norm(obs1['center'][:2] - obs2['center'][:2])
                if distance < max_dist:
                    cluster.append(obs2)
                    used.add(j)
                    
           
            centers = np.array([o['center'] for o in cluster])
            spans = np.array([o['span'] for o in cluster])
            merged.append({
                'center': np.mean(centers, axis=0),
                'span': np.max(spans) + 0.5*max_dist  
            })
            used.add(i)
        
        return merged

    def destroy(self):
        """Cleanup sensors"""
        if self.rgb_sensor:
            self.rgb_sensor.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
        self.rgb_sensor = None
        self.lidar_sensor = None

    def _process_rgb_image(self, image):
        
        self.rgb_timestamp = image.timestamp
    
        # Store vehicle transform at the time of image capture
        self.rgb_vehicle_transform = self.parent.get_transform()
        
        # Process image data...
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.rgb_image = array[:, :, :3]
        
        # Detect lane lines in this frame
        self.detect_lane_lines(self.rgb_image)
        
        
#################################################################################
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def calculate(self, error, dt=0.05):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
class SteeringPID:
    def __init__(self, kp=0.8, ki=0.01, kd=0.08):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt=0.05):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
#####################################################################################
class AutonomousController:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.target_speed = 15.0  # desired speed in km/h
        self.pid_throttle = PIDController(Kp=2.5, Ki=0.01, Kd=0.1)
        self.lane_shift_target = None  # target lane center for shifting
        self.lane_shift_tol = 0.1  # tolerance for lane shift completion in meters
        self.steering_pid = SteeringPID()
        self.lookahead_points = []

    def update(self):
       
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        control = carla.VehicleControl()
        error = self.target_speed - current_speed
        throttle = self.pid_throttle.calculate(error, dt=0.05)

        lane_data = self.world.semantic_camera.lane_boundaries
        obstacles = self.world.semantic_camera.obstacles
        vehicle_local_x = 0

        print(f"Obstacles detected count: {len(obstacles)}")
        #print(f"Obstacle details: {obstacles}")
        print(f"Lane boundaries: {lane_data}")
        vehicle_location = self.vehicle.get_location()
        #print(f"X: {vehicle_location.x}")

        
    
        l, r = None, None
        lane_centers = []
        free_lanes = []
        current_yaw = 0

        if not lane_data or len(lane_data) < 1:
            self.stop_and_horn()
            return
    
    # Calculate drivable area bounds
        try:
            left_bound = min(lane[0] for lane in lane_data if lane)
            right_bound = max(lane[0] for lane in lane_data if lane)
            lane_width = abs(right_bound - left_bound)
        except:
            self.stop_and_horn()
            return

        nearest_obstacle = None
        min_distance = float('inf')
        for obs in obstacles:
            # obs['center'] is in vehicle-local coordinates (x=lateral, y=forward)
            if 0 < obs['center'][1] < min_distance:  # Only consider front obstacles
                min_distance = obs['center'][1]
                nearest_obstacle = obs
        
        # Emergency braking system
        if nearest_obstacle and nearest_obstacle['center'][1] < 5.0:  
            control.throttle = 0.0
            control.brake = min(1.0, (10.0 - nearest_obstacle['center'][1])/5.0)
            self.vehicle.apply_control(control)
            return
        
        target_x = (left_bound + right_bound) / 2  # Center by default
    
    # Check for blocked lanes
        left_blocked = any(obs['center'][0] < (left_bound + right_bound)/2 and 
                        obs['center'][1] < 20 for obs in obstacles)
        right_blocked = any(obs['center'][0] >= (left_bound + right_bound)/2 and 
                        obs['center'][1] < 20 for obs in obstacles)
        
        # Lane shift logic
        if vehicle_local_x < (left_bound + right_bound)/2 and left_blocked:
            target_x = right_bound - lane_width/4  # Shift to right lane
        elif vehicle_local_x >= (left_bound + right_bound)/2 and right_blocked:
            target_x = left_bound + lane_width/4 
        

        lateral_error = target_x - vehicle_local_x
    
    # Steering control (improved PID)
        steer_output = self.steering_pid.compute(lateral_error)
        
        # Speed control (dynamic target speed)
        self.target_speed = 25.0 if lane_width > 3.5 else 15.0
        speed_error = self.target_speed - current_speed
        throttle_output = self.pid_throttle.calculate(speed_error)
        
        # Apply controls with smooth transitions
        control.steer = np.clip(steer_output, -0.8, 0.8)
        control.throttle = np.clip(throttle_output, 0.0, 0.8)
        control.brake = 0.0
        
        # Full stop if both lanes blocked
        if left_blocked and right_blocked:
            control.throttle = 0.0
            control.brake = 1.0
        
        self.vehicle.apply_control(control)
        

    def _calculate_desired_yaw(self, lookahead_points):
        """Calculate desired yaw based on lookahead points"""
        if not lookahead_points or all(p is None for p in lookahead_points):
            return 0.0  # assume forward

        target_point = next((p for p in lookahead_points if p is not None), None)
        dx, dy = target_point[0], target_point[1]
        target_yaw = math.degrees(math.atan2(dy, dx))
        
        return target_yaw


    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def get_free_lanes(self, obstacles, lane_centers):
        free_lanes = []
        for center in lane_centers:
            if not self.is_lane_blocked(obstacles, center):
                free_lanes.append(center)
        return free_lanes

    def is_lane_blocked(self, obstacles, lane_center):
        c=1
        for obstacle in obstacles:
            obs_x = obstacle['center'][0] 
            obs_y = obstacle['center'][1]  
            c+=1
           
            if obs_y > -5 and obs_y<20:  # In front of vehicle
                angle = math.degrees(math.atan2(obs_x, obs_y))
                #if -45 <= angle <= 115:  
                if abs(obs_x - lane_center) < obstacle['span']/2:
                    return True
        #print(c)
        return False

    def _generate_lookahead_path(self, num_points=4):
        path = []
        for i in range(num_points):
            distance = 8 * (1.5 ** i)  
            path.append(self._get_point_along_lane(distance))
        return path

    def _get_point_along_lane(self, distance):
        """Get point at specified distance along detected lane"""
        lane_line = self.world.semantic_camera.lane_boundaries[0]
        if lane_line is None or len(lane_line)==0:
            return None
        
        #print(lane_line)
        points = np.array([point for point in lane_line])
        #print("Sussy2 is not the convict")
        # Calculate cumulative distances
        points = [[points[0], points[1]],[points[2],points[3]]]
        diffs = np.diff(points, axis=0)
        dists = np.hypot(diffs[:,0], diffs[:,1])
        cum_dists = np.insert(np.cumsum(dists), 0, 0)
        
        if cum_dists[-1] == 0:
            return None
            
        # Find segment containing target distance
        target_dist = min(distance, cum_dists[-1])
        segment_idx = np.searchsorted(cum_dists, target_dist) - 1
        
        # Linear interpolation within segment
        seg_start = np.array(points[segment_idx])
        seg_end = np.array(points[segment_idx+1])
        seg_length = dists[segment_idx]

        
        frac = (target_dist - cum_dists[segment_idx]) / seg_length
        
        interpolation = seg_start + frac * (seg_end - seg_start)
        return interpolation
    
    def stop_and_horn(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        if hasattr(control, 'horn'):
            control.horn = True
        self.vehicle.apply_control(control)
        print("Vehicle stopped and horn activated.")
###############################################################################
class RGBCamera(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self.parent = parent_actor
        self.hud = hud
        self._setup_camera()

    def _setup_camera(self):
        """Setup the RGB camera for display"""
        world = self.parent.get_world()
        bp_library = world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')

        bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        bp.set_attribute('image_size_y', str(self.hud.dim[1]))
        bp.set_attribute('fov', '90')

        transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5))
        self.sensor = world.spawn_actor(bp, transform, attach_to=self.parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: RGBCamera._process_image(weak_self, image))

    
    @staticmethod
    def _process_image(weak_self, image):
        
        self = weak_self()
        if not self or not hasattr(self.parent, 'semantic_camera'):
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)).copy()
        array = array[:, :, :3]
        array = array[:, :, ::-1].copy()  # BGR to RGB
        line = []
        i=0
        
        if hasattr(self.parent.semantic_camera, 'lane_pixel_boundaries'):
            for lane in self.parent.semantic_camera.lane_pixel_boundaries:
                if len(lane) == 4:  # x1,y1,x2,y2
                    line.append([[lane[0], lane[1]], [lane[2], lane[3]]])
                    
                    i+=1
            if line:   

                points = np.array([line[0][0],line[0][1], line[1][0], line[1][1]], dtype=np.int32)
                hull = cv2.convexHull(points)
                overlay = array.copy()
                cv2.fillPoly(array, [hull], color=(0, 255, 0))
                cv2.addWeighted(array, 0.2, array, 0.8, 0)

        # Draw obstacles
        if hasattr(self.parent.semantic_camera, 'obstacles'):
            for obstacle in self.parent.semantic_camera.obstacles:
                center = obstacle['center']
                span = obstacle['span']
                
                # Project to image coordinates
                x = int((center[0] / center[2]) * 800 / (2 * math.tan(math.radians(90/2))) + image.width/2)
                y = int((center[1] / center[2]) * 800 / (2 * math.tan(math.radians(90/2))) + image.height/2)
                size = int(span * 30 / max(center[2], 1))  # Scale by distance
                
                if 0 <= x < image.width and 0 <= y < image.height:
                    cv2.circle(array, (x, y), size, (0, 0, 255), -1)
                    cv2.putText(array, f"{center[0]:.1f}m", 
                            (x-size, y-size-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    """
     def _draw_center_lines(self, image, drivable_area):
        
        if len(drivable_area) < 4:
            return

        # Split polygon into left and right boundaries
        half = len(drivable_area) // 2
        left_boundary = [self._world_to_screen(p) for p in drivable_area[:half]]
        right_boundary = [self._world_to_screen(p) for p in drivable_area[half:]]

        # Calculate center line points
        center_line = []
        for l, r in zip(left_boundary, reversed(right_boundary)):
            if l and r:
                cx = (l[0] + r[0]) // 2
                cy = (l[1] + r[1]) // 2
                center_line.append((cx, cy))

        # Draw center line
        if len(center_line) >= 2:
            cv2.polylines(image, [np.array(center_line)], 
                        isClosed=False, color=(0, 255, 0), thickness=2)

    """
    """

    def _overlay_drivable_area(self, image, drivable_area, alpha=0.3):
        if not drivable_area or len(drivable_area) < 4:
            return
        pts = [self._world_to_screen(point) for point in drivable_area if self._world_to_screen(point)]
        if len(pts) < 3:
            return
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        overlay = image.copy()
        # Fill polygon in green
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color=(0, 255, 0))
        # Draw outline in red
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)"""

    """
    def _overlay_obstacles(self, image, obstacles):
       # 
        if not obstacles:
            return
        
        obstacle_overlay = image.copy()
        
        # First draw lane boundaries with green color
        if hasattr(self, 'lane_boundaries') and self.lane_boundaries:
            for lane_boundary in self.lane_boundaries:
                if len(lane_boundary) >= 4:  # Has valid coordinates
                    x1, y1, x2, y2 = map(int, lane_boundary)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Now draw obstacles with distance information
        for obstacle in obstacles:
            center = obstacle['center']
            span = obstacle['span']
            
            # Project to camera view
            pt2d = self._world_to_screen(center)
            if not pt2d:
                continue  # Skip if not visible in camera frame
                
            # Calculate distance from camera
            camera_loc = self.sensor.get_transform().location
            obstacle_loc = carla.Location(x=center[0], y=center[1], z=center[2])
            distance = camera_loc.distance(obstacle_loc)
            
            # Scale size inversely with distance for realistic perspective
            size = int(span * 15 / max(distance/10, 1))
            size = max(10, min(size, 100))  # Size constraints
            
            # Draw obstacle with visual depth cues
            alpha = max(0.2, min(0.8, 10/distance))  # Opacity based on distance
            cv2.circle(obstacle_overlay, pt2d, size, (0, 0, 255), -1)  # Red fill
            
            # Add distance label
            cv2.putText(image, f"m", 
                    (pt2d[0]-size//2, pt2d[1]-size-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend the obstacle overlay with main image
        cv2.addWeighted(obstacle_overlay, 0.4, image, 0.6, 0, image)
    """
    """   
    def _world_to_screen(self, world_point):
       
        # Get camera transform
        cam_transform = self.sensor.get_transform()
        
        # Transform point to camera space
        if isinstance(world_point, carla.Location):
            point = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        elif isinstance(world_point, np.ndarray) or isinstance(world_point, list):
            point = np.array([world_point[0], world_point[1], world_point[2], 1.0])
        else:
            return None
            
        # Get camera matrix (extrinsic)
        world_to_camera = np.array(cam_transform.get_inverse_matrix())
        
        # Transform point to camera coordinates
        point_camera = np.dot(world_to_camera, point)
        
        # Camera intrinsic parameters
        image_w = self.hud.dim[0]
        image_h = self.hud.dim[1]
        fov = 90  # Default FOV for the camera
        
        # Build intrinsic matrix K
        focal = 800 / (2.0 * np.tan(np.radians(fov) / 2.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        
        # Project 3D->2D
        point_img = np.dot(K, point_camera[:3])
        
        # Normalize
        if point_img[2] <= 0:
            return None  # Point is behind camera
            
        x = int(point_img[0] / point_img[2])
        y = int(point_img[1] / point_img[2])
        
        # Check if within image bounds
        if 0 <= x < image_w and 0 <= y < image_h:
            return (x, y)
        return None"""
    """
    def _get_camera_matrix(self, transform):
        
        rotation = transform.rotation
        location = transform.location
        
        # Convert rotation to radians
        pitch = np.radians(rotation.pitch)
        yaw = np.radians(rotation.yaw)
        roll = np.radians(rotation.roll)
        
        # Create rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(pitch), -np.sin(pitch)],
                      [0, np.sin(pitch), np.cos(pitch)]])
        
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                      [np.sin(roll), np.cos(roll), 0],
                      [0, 0, 1]])
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        # Translation vector
        T = np.array([[location.x], [location.y], [location.z]])
        
        # Camera matrix
        return np.vstack((np.hstack((R, -T)), [0, 0, 0, 1]))
    """
    def render(self, display):
        if hasattr(self, 'rendered_image') and self.rendered_image is not None:
            # Convert to pygame surface
            rgb_image = self.rendered_image.swapaxes(0, 1)
            pygame_surface = pygame.surfarray.make_surface(rgb_image)
            display.blit(pygame_surface, (0, 0))
        elif self.surface is not None:
            display.blit(self.surface, (0, 0))


    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
#############################################################################################################
class TopdownMap:
    def __init__(self, world, width, height):
        self.world = world
        self.width = width
        self.height = height
        self.scale = 5  # meters per pixel
        self.surface = pygame.Surface((width, height))
        self.show_obstacles = False

    def render(self, display):
        self.surface.fill((0, 0, 0))
        
        # Draw ego vehicle
        pygame.draw.polygon(self.surface, (0, 0, 255), [
            (self.width//2, self.height//2 - 20),
            (self.width//2 - 10, self.height//2 + 10),
            (self.width//2, self.height//2 + 15),
            (self.width//2 + 10, self.height//2 + 10)
        ])
        
        # Draw lane boundaries if available
        if hasattr(self.world.semantic_camera, 'lane_boundaries'):
            for lane in self.world.semantic_camera.lane_boundaries:
                if len(lane) == 4:
                    x1 = self.width//2 + int(lane[0]/self.scale)
                    y1 = self.height//2 + int(lane[1]/self.scale)
                    x2 = self.width//2 + int(lane[2]/self.scale)
                    y2 = self.height//2 + int(lane[3]/self.scale)
                    pygame.draw.line(self.surface, (0, 255, 0), (x1,y1), (x2,y2), 2)

        self._draw_drivable_area()
        if self.show_obstacles and hasattr(self.world.semantic_camera, 'obstacles'):
            self._draw_obstacles()

        display.blit(self.surface, (display.get_width() - self.width, 
                                   display.get_height() - self.height))

    def _draw_drivable_area(self):
        
        if not hasattr(self.world.semantic_camera, 'lane_boundaries') or len(self.world.semantic_camera.lane_boundaries) < 2:
            return

        left_boundary = self.world.semantic_camera.lane_boundaries[0]
        right_boundary = self.world.semantic_camera.lane_boundaries[1]

        if len(left_boundary) != 4 or len(right_boundary) != 4:
            return  # Ensure both boundaries have valid points

        # Convert boundary points to top-down map coordinates
        def convert_point(x, y):
            return (
                int(x / self.scale) + self.width // 2,
                self.height // 2 - int(y / self.scale)
            )

        # Extract and convert points from left and right boundaries
        l_start = convert_point(left_boundary[0], left_boundary[1])
        l_end = convert_point(left_boundary[2], left_boundary[3])
        r_start = convert_point(right_boundary[0], right_boundary[1])
        r_end = convert_point(right_boundary[2], right_boundary[3])

        # Create polygon points (left start, left end, right end, right start)
        polygon_points = [l_start, l_end, r_end, r_start]

        # Draw filled polygon for drivable area
        pygame.draw.polygon(self.surface, (0, 200, 0), polygon_points)
        # Optionally draw boundaries
        pygame.draw.line(self.surface, (0, 255, 0), l_start, l_end, 2)
        pygame.draw.line(self.surface, (0, 255, 0), r_start, r_end, 2)

    def _draw_obstacles(self):
        for obstacle in self.world.semantic_camera.obstacles:
            obs_x = obstacle['center'][0] 
            obs_y = obstacle['center'][1]
            span = obstacle['span']
            
            # Convert to map coordinates
            px = int(obs_x/self.scale) + self.width//2
            py = self.height//2 - int(obs_y/self.scale)
            
            # Draw obstacle with actual dimensions
            size = int(span/self.scale)
            pygame.draw.circle(self.surface, (255,0,0), (px, py), max(2, size//2))
########################################################################################
class HUD(object):
    def __init__(self, width, height):  # Fixed constructor name
        self.dim = (width, height)
        # Initialize pygame font
        pygame.font.init()
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = []
        self.help = HelpText()
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self.vehicle_status = {}

    def notification(self, text, seconds=2.0):
        """Add a notification to the queue"""
        self._notifications.append((pygame.time.get_ticks(), text, seconds))

    def tick(self, world, clock):
        """Tick for HUD elements update"""
        self._notifications = [
            x for x in self._notifications if pygame.time.get_ticks() - x[0] < x[2] * 1000
        ]
        if world.player is not None:
            t = world.player.get_transform()
            v = world.player.get_velocity()
            c = world.player.get_control()
            self.vehicle_status = {
                'Location': (round(t.location.x, 1), round(t.location.y, 1), round(t.location.z, 1)),
                'Speed': round(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2), 1),  # Fixed speed calculation
                'Throttle': round(c.throttle, 2),
                'Steer': round(c.steer, 2),
                'Brake': round(c.brake, 2),
                'Reverse': c.reverse,
                'Recording': getattr(world.semantic_camera, 'recording', False)
            }

    def render(self, display):
        """Render the HUD"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))

            v_offset = 4
            bar_h_offset = 100
            bar_width = 106

            for item in self.vehicle_status.items():
                if v_offset + 18 > self.dim[1]:
                    break
                key, value = item

                # Boolean (checkbox-style)
                if isinstance(value, bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if value else 1)
                # Tuple (e.g., Location)
                elif isinstance(value, tuple):
                    value_str = ', '.join(map(str, value))
                    font_surface = self._font_mono.render(f"{key}: {value_str}", True, (255, 255, 255))
                    display.blit(font_surface, (8, v_offset))
                    v_offset += 26
                    continue
                # Number (draw bar)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = float(value) / 100.0 if isinstance(value, (float, int)) else 0
                    if key == 'Speed':
                        f = min(f, 1.0)  # Normalize speed to max 100 km/h for the bar
                    f = max(0.0, min(f, 1.0))
                    rect = pygame.Rect((bar_h_offset + 1, v_offset + 9), (int(f * (bar_width - 2)), 4))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                # Display the raw value as a string
                font_surface = self._font_mono.render(f"{key}: {value}", True, (255, 255, 255))
                display.blit(font_surface, (8, v_offset))
                v_offset += 26

            # Display help
            display.blit(self.help.render(), (0, self.dim[1] - self.help.height))

        # Display notifications
        for idx, notification in enumerate(self._notifications):
            notification_surface = pygame.Surface((280, 20), pygame.SRCALPHA)
            notification_surface.fill((0, 0, 0, 150))  # RGBA color with alpha
            notification_pos = (self.dim[0] - 300, 40 + 20 * idx)
            display.blit(notification_surface, notification_pos)
            display.blit(
                self._font_mono.render(notification[1], True, (255, 255, 255)),
                (self.dim[0] - 290, 40 + 20 * idx)
            )
#########################################################################################
class HelpText(object):
    def __init__(self):
        # Control tips
        self.text = [
            'W / Up : Throttle',
            'S / Down : Brake',
            'A / Left : Steer Left',
            'D / Right : Steer Right',
            'Q : Toggle Reverse',
            'Space : Handbrake',
            'P : Toggle Autopilot',
            'R : Toggle Recording',
            'ESC / Ctrl+Q : Quit',
        ]
        self.font = pygame.font.Font(pygame.font.get_default_font(), 14)
        self.height = len(self.text) * 22

    def render(self):
        """Render help text"""
        help_surface = pygame.Surface((600, self.height))
        help_surface.fill((0, 0, 0))
        help_surface.set_alpha(150)

        v_offset = 10
        for item in self.text:
            if v_offset + 18 > self.height:
                break
            font_surface = self.font.render(item, True, (255, 255, 255))
            help_surface.blit(font_surface, (10, v_offset))
            v_offset += 22

        return help_surface
####################################################################################
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot=False):  # Fixed constructor
        self._world = world
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl() if isinstance(world.player, carla.Vehicle) else carla.WalkerControl()
        self._steer_cache = 0.0
        self._lidar_visualization = True
        self._last_brake = 0.0
        self._max_steer = 0.5  # Reduced from original 0.7 to 0.5 for smoother steering
        self._steer_step = 0.1  # New steer increment value
        self._reverse_mode = False
        self.autonomous_controller = None
        

        # Initialize vehicle properly
        if isinstance(world.player, carla.Vehicle):
            world.player.set_autopilot(self._autopilot_enabled)
            if self._autopilot_enabled:
                self.autonomous_controller = AutonomousController(world.player, world)
        elif isinstance(world.player, carla.Walker):
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")

    def parse_events(self, client, world, clock):
        """Parse keyboard events with improved controls"""
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key in weather_keys:
                    preset_name = weather_keys[event.key]
                    apply_weather(world, preset_name)

            if event.type == pygame.KEYUP:
                # Emergency brake when releasing throttle
                if event.key in [K_UP, K_w]:
                    self._control.brake = min(self._last_brake + 0.2, 1.0)
                
                # Toggle features
                elif event.key == K_p:  # Better autopilot toggle
                    self._toggle_autopilot(world)
                elif event.key == K_r:  # Safer recording toggle
                    self._toggle_recording(world)
                elif event.key == K_l:  # Lidar visualization
                    self._toggle_lidar_visualization(world)
                elif event.key == K_q:
                    self._reverse_mode = not self._reverse_mode
                    world.hud.notification(f"Reverse mode {'ON' if self._reverse_mode else 'OFF'}")

        # Continuous key press handling
        keys = pygame.key.get_pressed()
        if not self._autopilot_enabled:
            self._handle_continuous_input(keys, clock, world)
        
        return False

    def _toggle_autopilot(self, world):
        if isinstance(world.player, carla.Vehicle):
            self._autopilot_enabled = not self._autopilot_enabled
            world.topdown_map.show_obstacles = self._autopilot_enabled

            if not self._autopilot_enabled:
                #self.awaiting_input = True
                self.autonomous_controller = AutonomousController(world.player, world)  # Instantiate your controller
                initial_control = carla.VehicleControl()
                initial_control.throttle = 0.0
                initial_control.brake = 0.0
                initial_control.steer = 0.0
                world.player.apply_control(initial_control)
            world.hud.notification(f"Autopilot {'ON' if self._autopilot_enabled else 'OFF'}")
    
    
    def _toggle_recording(self, world):
        if hasattr(world, 'semantic_camera'):
            is_recording = world.semantic_camera.toggle_recording()
            world.hud.notification(f"Recording {'ON' if is_recording else 'OFF'}")

    def _toggle_lidar_visualization(self, world):
        self._lidar_visualization = not self._lidar_visualization
        world.hud.notification(f"LiDAR Visualization {'ON' if self._lidar_visualization else 'OFF'}")

    def _handle_continuous_input(self, keys, clock, world):
        
        if isinstance(self._control, carla.VehicleControl):
            # Throttle/Brake with acceleration control
            target_speed = 0.0
            if keys[K_UP] or keys[K_w]:
                target_speed = 1.0
            elif keys[K_DOWN] or keys[K_s]:
                target_speed = -1.0

            # Smooth acceleration/deceleration
            if target_speed > 0:
                self._control.throttle = 1.0
                self._control.brake = 0.0
                self._control.reverse = self._reverse_mode
            elif target_speed < 0:
                self._control.throttle = 0.0
                self._control.brake = 1.0
            else:
                self._control.throttle = 0.0
                self._control.brake = 0.0

            # Steering with inertia
            steer_target = 0.0
            if keys[K_LEFT] or keys[K_a]:
                steer_target = -self._max_steer
            elif keys[K_RIGHT] or keys[K_d]:
                steer_target = self._max_steer

            # Smooth steering transition
            if abs(self._steer_cache - steer_target) > 0.01:
                self._steer_cache += (steer_target - self._steer_cache) * self._steer_step
            else:
    
                if abs(self._steer_cache) > 0.01:
                    self._steer_cache *= 0.5  # Dampen toward 0
                else:
                    self._steer_cache = 0.0

            self._control.steer = round(self._steer_cache, 2)

            # Handbrake
            self._control.hand_brake = keys[K_SPACE]

            # Apply control
            self._control.reverse = self._reverse_mode
            world.player.apply_control(self._control)

        elif isinstance(self._control, carla.WalkerControl):
            # Walker controls (improved movement)
            self._control.speed = 0.0
            if keys[K_UP] or keys[K_w]:
                self._control.speed = 3.0  # More reasonable walking speed
            if keys[K_LEFT] or keys[K_a]:
                self._rotation.yaw -= 1.0
            if keys[K_RIGHT] or keys[K_d]:
                self._rotation.yaw += 1.0

            self._control.jump = keys[K_SPACE]
            self._control.direction = self._rotation.get_forward_vector()
            world.player.apply_control(self._control)
#####################################################################################
class World(object):
    def __init__(self, carla_world, hud, args):  # Fixed constructor name
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.rgb_camera = None
        self.topdown_map = None
        self.semantic_camera = None
        self.args = args
        self._actor_filter = args.filter
        self.output_dir = args.output_dir
        self.restart()

    def restart(self):
    
        # Cleanup existing actors
        self.destroy()

        # Get blueprint with collision checks
        blueprint = self.world.get_blueprint_library().find(self._actor_filter)

        if not blueprint:

            raise ValueError(f"Actor filter '{self._actor_filter}' not found")

        blueprint.set_attribute('role_name', 'hero')
        
        # Try all spawn points systematically
        spawn_points = self.world.get_map().get_spawn_points()
        for idx, spawn_point in enumerate(spawn_points):
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            if self.player:
                print(f"Spawned at point {idx}: {spawn_point.location}")
                break
        else:
            raise RuntimeError(f"Failed to spawn player at all {len(spawn_points)} points")

        # Critical: Wait for spawn to complete
        self.world.tick()
        time.sleep(0.5)  # Allow physics to settle

        # Initialize sensors
        self._init_sensors()

    def tick(self, clock):
        """Improved tick with error handling"""
        try:
            self.hud.tick(self, clock)
            self.world.tick()
            
        except RuntimeError as e:
            print(f"Simulation tick failed: {e}")

    def render(self, display):
        """Safer rendering with null checks"""
        try:
            if self.rgb_camera:
                self.rgb_camera.render(display)
            if self.topdown_map:
                self.topdown_map.render(display)
            if self.hud:
                self.hud.render(display)
        except Exception as e:
            print(f"Rendering error: {e}")

    def destroy(self):
   
        if self.rgb_camera:
            self.rgb_camera.destroy()
        if self.semantic_camera:
            self.semantic_camera.destroy()  # Now has destroy() method
        if self.player:
            self.player.destroy()
        # Reset references
        self.rgb_camera = None
        self.semantic_camera = None
        self.player = None

    def _init_sensors(self):
    
        try:
            self.semantic_camera = SemanticCamera(
                self.player, 
                self.output_dir,
                hud=self.hud
            )
            self.player.semantic_camera = self.semantic_camera
            self.rgb_camera = RGBCamera(self.player, self.hud)
            self.topdown_map = TopdownMap(self, 200, 200)
            self.player.autonomous_controller = AutonomousController(self.player, self)
        except Exception as e:
            print(f"Sensor initialization failed: {e}")
            self.destroy()
            raise
    # In main game loop setup:

weather_keys = {
    pygame.K_1: "ClearNoon",
    pygame.K_2: "CloudySunset",
    pygame.K_3: "WetCloudyNoon",
    pygame.K_4: "MidRainyNoon",
    pygame.K_5: "HardRainNoon",
    pygame.K_6: "SoftRainSunset",
    pygame.K_7: "Snowy"
}

weather_presets = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "Snowy": carla.WeatherParameters(
    precipitation = 90,          # Intensity of "snowfall" (0-100)
    precipitation_deposits = 90, # Snow accumulation on surfaces
    cloudiness = 90,             # Overcast sky
    wind_intensity = 50,         # Wind strength
    fog_density = 60,            # Reduced visibility
    fog_distance = 10,           # How far fog starts
    wetness = 80,                # Wet surfaces (mimics snow melting)
    sun_altitude_angle = 15      # Low winter sun angle
)
}



def apply_weather(world, preset_name):
    if preset_name in weather_presets:
        world.world.set_weather(weather_presets[preset_name])
        print(f"[Weather] Changed to: {preset_name}")
    else:
        print(f"[Weather] Unknown preset: {preset_name}")
#############################################################################
def game_loop(args):
    """Main loop for the CARLA autonomous driving simulation"""

    # Initialize Pygame
    pygame.init()
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("CARLA Autonomous Driving Simulator")
    
    # Connect to CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    print(f"Connected to CARLA server: {args.host}:{args.port}")
    
    try:
        # Create HUD and World
        hud = HUD(args.width, args.height)
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        world = World(client.get_world(), hud, args)
        
        

        
        controller = KeyboardControl(world, start_in_autopilot=False)
        

        # Create clock for FPS control
        clock = pygame.time.Clock()
        
        # Main loop
        while True:
            try:
                # Tick the world
                clock.tick_busy_loop(60)  # Target 60 FPS
                world.tick(clock)

                if controller._autopilot_enabled and controller.autonomous_controller is not None:
                    controller.autonomous_controller.update()
                    #print("Updated!")
                
                # Render world state
                world.render(display)
                pygame.display.flip()
                
                # Process control inputs
                if controller.parse_events(client, world, clock):
                    break
                    
                # Update HUD
                hud.server_fps = clock.get_fps()
                hud.simulation_time = world.world.get_snapshot().timestamp.platform_timestamp
                
            except RuntimeError as e:
                print(f"Simulation error: {e}")
                break
                
    except Exception as e:
        print(f"Fatal error: {e}")
        
    finally:
        # Cleanup
        try:
            if 'world' in locals() and world:
                world.destroy()
        except Exception as e:
            print(f"Cleanup error: {e}")
#####################################################################################################
def main():
    """Main function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Autonomous Driving with Lane Detection and Obstacle Avoidance')
    
    
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '--output-dir',
        metavar='DIR',
        default='./output',
        help='output directory (default: ./output)')
    argparser.add_argument(
        '--width',
        default=800,
        type=int,
        help='window width (default: 800)')
    argparser.add_argument(
        '--height',
        default=600,
        type=int,
        help='window height (default: 600)')
    argparser.add_argument(
        '--lidar-range',
        default=50.0,
        type=float,
        help='LiDAR detection range in meters (default: 50.0)')

    args = argparser.parse_args()

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'Error occurred: {str(e)}')

if __name__ == "__main__":
    main()