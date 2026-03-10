import time
import os
import numpy as np
import utils
from simulation import vrep
import trimesh
import cv2

class Robot(object):
    def __init__(self, stage, goal_object, obj_mesh_dir, num_obj, workspace_limits,
                 is_testing, test_preset_cases, test_preset_file,
                 goal_conditioned, grasp_goal_conditioned):

        # Core configuration
        self.workspace_limits = workspace_limits
        self.num_obj = num_obj
        self.stage = stage
        self.goal_conditioned = goal_conditioned
        self.grasp_goal_conditioned = grasp_goal_conditioned
        self.goal_object = goal_object

        # Define colors for object meshes (Tableau palette)

        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purpl
                                        [118, 183, 178], # cyan
                                        [255, 157, 167], # pink
                                        [58.0, 100.0, 140.0], # blue2
                                        [140, 107, 70], # brown2
                                        [220, 122, 23], # orange2
                                        [207.0, 160, 52], # yellow2
                                        [20, 255, 145], # grass green 2
                                        [130, 18, 17], # dark red 2
                                        [42, 41, 255], #blue 3,
                                        [245, 222, 179], #wheat
                                        [41,132,132], #cyan2
                                        [193, 50, 255], #purple2
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purpl
                                        [118, 183, 178], # cyan
                                        [255, 157, 167], # pink
                                        [58.0, 100.0, 140.0], # blue2
                                        [140, 107, 70]] # brown2
                                       )/255.0 # gray2

        # Mark goal object color as greenish to highlight
        green_color = [89.0/255.0, 161.0/255.0, 79.0/255.0]
        self.color_space = np.insert(self.color_space, self.goal_object, green_color, axis=0)
        
        color_avg = [[67,68,103,104,142,144],   #blue
                            [132,133,100,101,81,82], #brown
                            [207,211,121,124,37,38], #orange
                            [203,205,172,174,62,63], #yellow
                            [158,160,149,150,146,148], #grey
                            [218,222,75,76,76,78], #red
                            [148,151,103,105,136,139], #purple
                            [101,103,155,158,151,154], #cyan
                            [217,219,132,133,142,143], #pink
                            [50,51,85,87,120,122], #blue2
                            [120,121,91,92,60,61], #brown2
                            [188,190,104,106,20,20], #orange2
                            [175,178,135,138,44,45], #yellow2
                            [17,17,218,220,124,126], #grass green2
                            [111,113,15,15,15,15], #dark red 2
                            [36,37,35,36,218,221], #blue3
                            [209,211,189,191,152,154], #wheat
                            [35,35,113,115,113,115], #cyan2
                            [163,165,43,43,217,219], #purple2
                            [132,133,100,101,81,82], #brown
                            [207,211,121,124,37,38], #orange
                            [203,205,172,174,62,63], #yellow
                            [158,160,149,150,146,148], #grey
                            [218,222,75,76,76,78], #red
                            [148,151,103,105,136,139], #purple
                            [101,103,155,158,151,154], #cyan
                            [217,219,132,133,142,143], #pink
                            [50,51,85,87,120,122], #blue2
                            [120,121,91,92,60,61] #brown2
                            ]
        green_avg = [76,78,138,140,67,69]
        green_std = [3,5,3]
        color_std = [[2.5,4,6],   #blue
                                    [4.5,3,2.5], #brown
                                    [5.5,4,1.5], #orange
                                    [6,5,2], #yellow
                                    [5,5,4.5], #grey
                                    [7,2.5,2.5], #red
                                    [6.5,4,5.5], #purple
                                    [3.5,5,5], #cyan
                                    [4,2.5,2], #pink
                                    [2,4,5], #blue2
                                    [3,2,2], #brown2
                                    [5,3,1], #orange2
                                    [4,3,1], #yellow2
                                    [1,6,3], #grass green2
                                    [4,1,1], #dark red 2
                                    [1,1,6], #blue3
                                    [5,4.5,4], #wheat
                                    [1,3,3], #cyan2
                                    [5,1,5], #purple2
                                    [4.5,3,2.5], #brown
                                    [5.5,4,1.5], #orange
                                    [6,5,2], #yellow
                                    [5,5,4.5], #grey
                                    [7,2.5,2.5], #red
                                    [6.5,4,5.5], #purple
                                    [3.5,5,5], #cyan
                                    [4,2.5,2], #pink
                                    [2,4,5], #blue2
                                    [3,2,2] #brown2
                                    ]
        
        color_avg.insert(self.goal_object, green_avg)
        color_avg = np.array(color_avg)
        color_std.insert(self.goal_object, green_std)
        color_std = np.array(color_std)
        # Pre-compute color thresholds (lower/upper for each channel)
        color_threshold = np.zeros(np.shape(color_avg))
        for palette_index in range(20):
            for ch in range(6):
                if ch % 2 == 0:
                    color_threshold[palette_index, ch] = color_avg[palette_index, ch] - 3 * color_std[palette_index, ch // 2]
                else:
                    color_threshold[palette_index, ch] = color_avg[palette_index, ch] + 3 * color_std[palette_index, ch // 2]
        self.color_threshold = color_threshold
        
        # Read files in object mesh directory
        self.obj_mesh_dir = obj_mesh_dir
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        # Pre-draw random mesh indices and colors for this scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)), :]

        # Connect to simulator
        # Connect to simulator
        vrep.simxFinish(-1)  # Close any existing connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        self.is_testing = is_testing
        self.test_preset_cases = test_preset_cases
        self.test_preset_file = test_preset_file

        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # If testing, read object meshes and poses from test case file
        if self.is_testing and self.test_preset_cases:
            with open(self.test_preset_file, 'r') as f:
                lines = f.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for object_idx in range(self.num_obj):
                tokens = lines[object_idx].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, tokens[0]))
                self.test_obj_mesh_colors.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                self.test_obj_positions.append([float(tokens[4]), float(tokens[5]), float(tokens[6])])
                self.test_obj_orientations.append([float(tokens[7]), float(tokens[8]), float(tokens[9])])

        # Add objects to simulation environment
        self.add_objects()


    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):
        """Add each object to the scene with randomized pose or from preset."""

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        obj_number = len(self.obj_mesh_ind)
        
        print('object number:')
        print(obj_number)
        for object_idx in range(obj_number):
            curr_mesh_file = self._select_mesh_for_index(object_idx)
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            object_position = self._compute_drop_position()
            object_orientation = self._compute_random_orientation()
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1] + 0.1, self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = self._import_shape(curr_mesh_file, curr_shape_name, object_position, object_orientation, object_color)
            try:
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                if not (self.is_testing and self.test_preset_cases):
                    time.sleep(0.5)
            except:
                print("curr_shape_handle out of range problem")
                self.restart_sim()
                self.add_objects()


        self.prev_obj_positions = []
        self.obj_positions = []

    def _select_mesh_for_index(self, object_idx):
        return os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

    def _compute_drop_position(self):
        centered_ratio = 0.95
        xrange = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2)
        yrange = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2)
        drop_x = (1 - centered_ratio) * xrange * np.random.random_sample() + self.workspace_limits[0][0] + 0.1 + xrange * centered_ratio * 0.5
        drop_y = (1 - centered_ratio) * yrange * np.random.random_sample() + self.workspace_limits[1][0] + 0.1 + yrange * centered_ratio * 0.5
        return [drop_x, drop_y, 0.1]

    def _compute_random_orientation(self):
        return [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]

    def _import_shape(self, mesh_file, shape_name, position, orientation, color):
        return vrep.simxCallScriptFunction(
            self.sim_client,
            'remoteApiCommandServer',
            vrep.sim_scripttype_childscript,
            'importShape',
            [0,0,255,0],
            position + orientation + color,
            [mesh_file, shape_name],
            bytearray(),
            vrep.simx_opmode_blocking
        )



    def restart_sim(self):
        print('Restart simluation!')
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        # Workaround for simulator quirk: ensure gripper settles
        while gripper_position[2] > 0.4:
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):
        """Check if gripper is within workspace, as a proxy for sim stability."""
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        within_x = self.workspace_limits[0][0] - 0.1 < gripper_position[0] < self.workspace_limits[0][1] + 0.1
        within_y = self.workspace_limits[1][0] - 0.1 < gripper_position[1] < self.workspace_limits[1][1] + 0.1
        within_z = self.workspace_limits[2][0] < gripper_position[2] < self.workspace_limits[2][1]
        return within_x and within_y and within_z

    def get_obj_positions(self):
        """Return positions of all currently spawned objects."""
        positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            positions.append(object_position)
        return positions

    def mask(self, img, ob_id):
        """Binary mask for a specific object id based on RGB thresholds."""
        threshold = self.color_threshold[ob_id]
        ch1 = (img[:, :, 0] > threshold[0]) & (img[:, :, 0] < threshold[1])
        ch2 = (img[:, :, 1] > threshold[2]) & (img[:, :, 1] < threshold[3])
        ch3 = (img[:, :, 2] > threshold[4]) & (img[:, :, 2] < threshold[5])
        mask = (ch1 & ch2 & ch3).astype(np.uint8) * 255
        return mask


    def mask_all_obj(self, img):
        """Coarse mask for all objects based on grayscale threshold."""
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masks = np.full_like(img_grey, 255)
        masks[img_grey <= 55] = 0
        return masks



    def obj_contour(self, obj_ind):
        maxAttemptsToGetPosition = 3
        for attemp in range(0, maxAttemptsToGetPosition):
            try:
                # Get object pose in simulation
                sim_ret, obj_position = vrep.simxGetObjectPosition(self.sim_client, self.object_handles[obj_ind], -1, vrep.simx_opmode_blocking)
                sim_ret, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.object_handles[obj_ind], -1, vrep.simx_opmode_blocking)
                break
            except:
                time.sleep(3)
                print("Failed to Get handle camera and Get camera pose and intrinsics in simulation from Coppelia, remaining attempts times in get_obj_mask Function", maxAttemptsToGetPosition-attemp)

        obj_pose = self._compose_object_pose(obj_position, obj_orientation)
        obj_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[obj_ind]])
        mesh = trimesh.load_mesh(obj_mesh_file)
        # transform the mesh to world frame
        if obj_mesh_file.split('/')[-1] in ('2.obj', '6.obj'):
            mesh.apply_transform(obj_pose)
        else:
            swap_transform = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
            mesh.apply_transform(swap_transform)
            mesh.apply_transform(obj_pose)
        return mesh.vertices[:, 0:2]

    def _compose_object_pose(self, position, orientation_euler):
        obj_trans = np.eye(4,4)
        obj_trans[0:3,3] = np.asarray(position)
        euler = [orientation_euler[0], orientation_euler[1], orientation_euler[2]]
        obj_rotm = np.eye(4,4)
        obj_rotm[0:3,0:3] = utils.obj_euler2rotm(euler)
        return np.dot(obj_trans, obj_rotm)


    def get_camera_data(self):
        maxAttemptsToGetPosition = 4
        for attemp in range(0, maxAttemptsToGetPosition):
            try:
                # Get color image from simulation
                sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
                color_img = self._convert_raw_color(raw_image, resolution)

                # Get depth image from simulation
                sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
                depth_img = self._convert_raw_depth(depth_buffer, resolution)
                break
            except:
                time.sleep(10)
                print("Failed to Get get_camera_data", maxAttemptsToGetPosition-attemp)

        return color_img, depth_img

    def _convert_raw_color(self, raw_image, resolution):
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        return color_img.astype(np.uint8)

    def _convert_raw_depth(self, depth_buffer, resolution):
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        return depth_img * (zFar - zNear) + zNear

    def close_gripper(self, asynch=False):

        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        gripper_fully_closed = False

        for i in range(1000):
            if gripper_joint_position <= -0.045:
                print('the gripper is fully closed in %.2f seconds'%((i+1) * 0.01))
                gripper_fully_closed = True
                break
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            time.sleep(0.01)
            if new_gripper_joint_position >= gripper_joint_position:
                gripper_fully_closed = False
                break
            gripper_joint_position = new_gripper_joint_position
            if i == 999:
                print('closed girpper is wrong')
                print(gripper_joint_position)
        

        return gripper_fully_closed


    def open_gripper(self, asynch=False):

        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        for i in range(1000):
            if gripper_joint_position >= 0.03:
                print('the gripper is opened in %.2f seconds'%((i+1) * 0.01))
                return True
            else:
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                time.sleep(0.01)
            if(i == 999):
                print('open girpper is wrong')
                print(gripper_joint_position)
                return False
    


    def move_to(self, tool_position, tool_orientation):

        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02*move_direction/move_magnitude if move_magnitude != 0 else np.asarray([0.0,0.0,0.0])
        num_move_steps = int(np.floor(move_magnitude/0.02)) if move_magnitude != 0 else 0

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)



    # Primitives ----------------------------------------------------------
    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)
        # Move gripper to location above grasp target
        temp_position = np.asarray(self.get_obj_positions())
        temp_position = temp_position[:,2]
        temp_max_z_position = max(temp_position)
        grasp_location_margin = 0.1 + temp_max_z_position

        location_above_grasp_target = (position[0], position[1],  grasp_location_margin)
        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        if move_step[0] != 0:
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
        else:
            num_move_steps = 0
        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))
        #print('the first step is finish')
        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        gripper_success_open = self.open_gripper()
        
        if(gripper_success_open):
            pass
        else:
            return False, None, None, None, None, None, gripper_success_open

        # Approach grasp target
        self.move_to(position, None)
        # Get images before grasping
        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration
        # Get heightmaps before grasping
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics,
                                                                self.cam_pose, workspace_limits,
                                                                0.002)  # heightmap resolution from args
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()
        # Move gripper to location above grasp target
        self.move_to(location_above_grasp_target, None)
        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed
        print('After grasp, grasp success: %r' %(grasp_success))
        # Move the grasped object elsewhere
        if grasp_success:
            object_positions = np.asarray(self.get_obj_positions())
            object_positions = object_positions[:,2]
            grasped_object_ind = np.argmax(object_positions)
            grasped_object_handle = self.object_handles[grasped_object_ind]
            vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

            return grasp_success, color_img, depth_img, color_heightmap, valid_depth_heightmap, grasped_object_ind, gripper_success_open
        else:
            return grasp_success, None, None, None, None, None, gripper_success_open


    def push(self, position, heightmap_rotation_angle, workspace_limits):

        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Adjust pushing point to be on tip of finger
        position[2] = position[2] + 0.012

        # Compute pushing direction
        push_orientation = [1.0,0.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # Move gripper to location above pushing point
        pushing_point_margin = 0.06
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude if move_magnitude != 0 else np.asarray([0.0,0.0,0.0])
        num_move_steps = int(np.floor(move_direction[0]/move_step[0])) if move_magnitude != 0 and move_step[0] != 0 else 0

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is closed
        self.close_gripper()

        # Approach pushing point
        self.move_to(position, None)

        # Compute target location (push to the right)
        push_length = 0.13
        target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
        push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

        # Move in pushing direction towards target location
        self.move_to([target_x, target_y, position[2]], None)

        # Move gripper to location above grasp target
        self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

        push_success = True
        return push_success


