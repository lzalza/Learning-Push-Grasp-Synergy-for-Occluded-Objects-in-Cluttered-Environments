import time
import os
import sys
import threading
import argparse
import numpy as np
import cv2
import torch
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from skimage.morphology.convex_hull import convex_hull_image
from scipy.ndimage.morphology import binary_dilation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Parse command line arguments
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,
                        help='number of objects to add to simulation')
arg_parser.add_argument('--stage', dest='stage', action='store', default='grasp_only',
                        help='stage of training: 1.grasp_only, 2.push_only, 3.push_grasp')
arg_parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,
                        help='use prioritized experience replay?')
arg_parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
arg_parser.add_argument('--grasp_reward_threshold', dest='grasp_reward_threshold', type=float, action='store', default=1.8)
arg_parser.add_argument('--grasp_explore', dest='grasp_explore', action='store_true', default=False)

arg_parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
arg_parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=20,
                        help='maximum number of test runs per case/scenario')
arg_parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
arg_parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

arg_parser.add_argument('--goal_obj_idx', dest='goal_obj_idx', type=int, action='store', default=2)
arg_parser.add_argument('--goal_conditioned', dest='goal_conditioned', action='store_true', default=False)
arg_parser.add_argument('--grasp_goal_conditioned', dest='grasp_goal_conditioned', action='store_true', default=False)

# Pre-loading and logging configuration
arg_parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
arg_parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
arg_parser.add_argument('--load_explore_snapshot', dest='load_explore_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
arg_parser.add_argument('--explore_snapshot_file', dest='explore_snapshot_file', action='store')
arg_parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
arg_parser.add_argument('--logging_directory', dest='logging_directory', action='store')
arg_parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,
                        help='save visualizations of FCN predictions?')
arg_parser.add_argument('--alternating_training', dest='alternating_training', action='store_true', default=False)
arg_parser.add_argument('--cooperative_training', dest='cooperative_training', action='store_true', default=False)

# Runtime control
arg_parser.add_argument('--max_iterations', dest='max_iterations', type=int, action='store', default=1001,
                        help='maximum number of training/testing iterations')

config = arg_parser.parse_args()


# Configuration parameters
object_count = config.num_obj
mesh_directory = os.path.abspath('objects/blocks')
heightmap_pixel_size = 0.002
np.random.seed(1234)
use_cpu_only = False

# Workspace boundaries
workspace_bounds = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

# Training configuration
training_stage = config.stage
max_push_episode_length = 5
grasp_reward_threshold = config.grasp_reward_threshold
alternating_training = config.alternating_training
cooperative_training = config.cooperative_training

# Q-learning hyperparameters
future_reward_discount = 0.5
use_experience_replay = config.experience_replay
decay_exploration_rate = config.explore_rate_decay

# Testing configuration
is_testing = config.is_testing
target_object_idx = config.goal_obj_idx
max_test_trials = 20
test_preset_cases = config.test_preset_cases
test_preset_file = os.path.abspath(config.test_preset_file) if test_preset_cases else None

# Model loading and logging configuration
load_snapshot = config.load_snapshot
snapshot_file = os.path.abspath(config.snapshot_file) if load_snapshot else None
continue_logging = config.continue_logging

if config.logging_directory is None:
    log_directory = os.path.abspath(config.logging_directory) if continue_logging else os.path.abspath('logs')
else:
    log_directory = config.logging_directory
save_visualizations = config.save_visualizations

# Goal-conditioned exploration options
load_explore_snapshot = config.load_explore_snapshot
explore_snapshot_file = os.path.abspath(config.explore_snapshot_file) if load_explore_snapshot else None

max_iterations = config.max_iterations

goal_conditioned = config.goal_conditioned
grasp_goal_conditioned = config.grasp_goal_conditioned

# Initialize system components
robot_agent = Robot(training_stage, target_object_idx, mesh_directory, object_count, workspace_bounds,
                     is_testing, test_preset_cases, test_preset_file,
                     goal_conditioned, grasp_goal_conditioned)

model_trainer = Trainer(training_stage, future_reward_discount,
                        is_testing, load_snapshot, snapshot_file,
                        load_explore_snapshot, explore_snapshot_file,
                        alternating_training, cooperative_training,
                        use_cpu_only, grasp_goal_conditioned)

data_logger = Logger(continue_logging, log_directory)
data_logger.save_heightmap_info(workspace_bounds, heightmap_pixel_size)

if continue_logging:
    model_trainer.preload(data_logger.transitions_directory)

# Initialize tracking variables
stagnation_counter = [2, 2] if not is_testing else [0, 0]
grasp_exploration_probability = 0.8 if not is_testing else 0.0
grasp_explore = config.grasp_explore

shared_state = {'executing_action': False,
                'primitive_action': None,
                'best_pix_ind': None,
                'push_success': False,
                'grasp_success': False,
                'grasp_reward': 0,
                'improved_grasp_reward': 0,
                'push_step': 0,
                'goal_obj_idx': config.goal_obj_idx,
                'goal_catched': 0,
                'border_occupy_ratio': 1,
                'decreased_occupy_ratio': 0,
                'restart_scene': 0,
                'episode': 0,
                'new_episode_flag': 0,
                'push_predictions': np.zeros((16, 224, 224), dtype=float),
                'grasp_predictions': np.zeros((16, 224, 224), dtype=float),
                'current_goal_idx': config.goal_obj_idx}

if continue_logging:
    if not is_testing:
        shared_state['episode'] = model_trainer.episode_log[len(model_trainer.episode_log) - 1][0]
    if training_stage == 'push_only':
        shared_state['push_step'] = model_trainer.push_step_log[model_trainer.iteration - 1][0]

robot_agent.restart_sim()


def check_object_out_of_bounds(goal_idx, workspace_bounds, heightmap_resolution):
    """Check if goal object is completely outside the scene."""
    object_contour = robot_agent.obj_contour(goal_idx)
    object_contour = utils.get_goal_coordinates(object_contour, workspace_bounds, heightmap_resolution)
    return (np.max(object_contour[:, 0]) < 0 or np.max(object_contour[:, 1]) < 0 or
            np.min(object_contour[:, 0]) > 224 or np.min(object_contour[:, 1]) > 224)


def apply_masks_to_predictions(grasp_pred, push_pred, color_heightmap, current_goal_idx):
    """Apply object masks to predictions."""
    target_mask = np.float32(robot_agent.mask(color_heightmap, current_goal_idx))
    masked_grasp_pred = np.multiply(grasp_pred, target_mask) / 255
    
    mask_weight = 0.05
    all_objects_mask = np.float32(robot_agent.mask_all_obj(color_heightmap))
    masked_push_pred = (np.multiply(push_pred, all_objects_mask) / 255 * (1 - mask_weight) +
                        np.multiply(push_pred, target_mask) / 255 * mask_weight)
    
    return masked_grasp_pred, masked_push_pred


def select_action_type(stage, push_conf, grasp_conf, push_step, border_ratio):
    """Determine which action type to execute based on stage and confidence scores."""

    if stage == 'grasp_only':
        action_type = 'grasp'
    elif stage == 'push_only':
        if grasp_conf > grasp_reward_threshold or push_step == max_push_episode_length:
            action_type = 'grasp'
        else:
            action_type = 'push'
    elif stage == 'push_grasp':
        if border_ratio > 0.08 or grasp_conf < 1.8 or 1.5 * push_conf > grasp_conf:
            action_type = 'push'
        else:
            action_type = 'grasp'
    
    return action_type


def compute_action_position(best_pixel_ind, depth_heightmap, workspace_bounds, heightmap_resolution, action_type):
    """Compute 3D position for action execution."""
    rotation_idx = best_pixel_ind[0]
    pixel_y = best_pixel_ind[1]
    pixel_x = best_pixel_ind[2]
    
    rotation_angle = np.deg2rad(rotation_idx * (360.0 / model_trainer.model.num_rotations))
    
    action_position = [pixel_x * heightmap_resolution + workspace_bounds[0][0],
                       pixel_y * heightmap_resolution + workspace_bounds[1][0],
                       depth_heightmap[pixel_y][pixel_x] + workspace_bounds[2][0]]
    
    if action_type == 'push':
        finger_width = 0.02
        safe_kernel_size = int(np.round((finger_width / 2) / heightmap_resolution))
        y_min = max(pixel_y - safe_kernel_size, 0)
        y_max = min(pixel_y + safe_kernel_size + 1, depth_heightmap.shape[0])
        x_min = max(pixel_x - safe_kernel_size, 0)
        x_max = min(pixel_x + safe_kernel_size + 1, depth_heightmap.shape[1])
        local_region = depth_heightmap[y_min:y_max, x_min:x_max]
        
        if local_region.size == 0:
            safe_z_position = workspace_bounds[2][0] - 0.01
        else:
            safe_z_position = np.max(local_region) + workspace_bounds[2][0] - 0.01
        action_position[2] = safe_z_position
    
    return action_position, rotation_angle


def prepare_push_only_metrics(grasp_conf, push_step, color_heightmap, depth_heightmap,
                               grasp_predictions, current_goal_mask_heightmap):
    """Prepare metrics before pushing in push_only stage."""
    if grasp_conf <= grasp_reward_threshold and push_step < max_push_episode_length:
        object_masks = []
        for obj_id in range(object_count):
            object_masks.append(np.float32(robot_agent.mask(color_heightmap, obj_id)))
        
        obj_grasp_predictions = utils.get_obj_grasp_predictions(grasp_predictions, object_masks, object_count)
        pre_push_single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
        print('pre-push grasp rewards: ', pre_push_single_predictions)
        
        pre_push_occupy_ratio = utils.get_occupy_ratio(current_goal_mask_heightmap, depth_heightmap)
        return pre_push_single_predictions, pre_push_occupy_ratio
    return None, None


def compute_improved_grasp_reward(grasp_predictions, latest_color_heightmap, latest_depth_heightmap,
                                   pre_push_single_predictions, current_goal_idx):
    """Compute improved grasp reward after pushing."""
    object_masks = []
    for obj_id in range(object_count):
        object_masks.append(np.float32(robot_agent.mask(latest_color_heightmap, obj_id)))
    
    obj_grasp_predictions = utils.get_obj_grasp_predictions(grasp_predictions, object_masks, object_count)
    post_push_single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
    print('reward of grasping after pushing: ', post_push_single_predictions)
    
    improved_rewards = [post_push_single_predictions[i] - pre_push_single_predictions[i]
                        for i in range(len(post_push_single_predictions))]
    print('expected reward of pushing(improved grasp reward)', improved_rewards)
    
    if not grasp_goal_conditioned:
        return np.max(improved_rewards)
    else:
        return improved_rewards[current_goal_idx]


def execute_push_action(action_position, rotation_angle, push_step, grasp_conf, pre_push_predictions=None, pre_occupy_ratio=None):
    """Execute push action and update state."""
    push_result = robot_agent.push(action_position, rotation_angle, workspace_bounds)
    print('Push result: %r' % push_result)
    model_trainer.grasp_obj_log.append([-1])
    data_logger.write_to_log('grasp-obj', model_trainer.grasp_obj_log)
    
    if training_stage == 'push_only':
        if grasp_conf <= grasp_reward_threshold and push_step < max_push_episode_length:
            latest_color_img, latest_depth_img = robot_agent.get_camera_data()
            latest_depth_img = latest_depth_img * robot_agent.cam_depth_scale
            
            latest_color_heightmap, latest_depth_heightmap = utils.get_heightmap(
                latest_color_img, latest_depth_img, robot_agent.cam_intrinsics,
                robot_agent.cam_pose, workspace_bounds, heightmap_pixel_size)
            
            latest_goal_mask_heightmap = robot_agent.mask(latest_color_heightmap, shared_state['current_goal_idx'])
            latest_goal_mask_heightmap = np.float32(latest_goal_mask_heightmap)
            
            if pre_push_predictions is not None:
                improved_reward = compute_improved_grasp_reward(
                    shared_state['grasp_predictions'], latest_color_heightmap, latest_depth_heightmap,
                    pre_push_predictions, shared_state['current_goal_idx'])
                
                if not grasp_goal_conditioned:
                    shared_state['improved_grasp_reward'] = improved_reward
                else:
                    shared_state['improved_grasp_reward'] = improved_reward
                print('thread improved grasp reward:', shared_state['improved_grasp_reward'])
            
            if pre_occupy_ratio is not None:
                occupy_ratio = utils.get_occupy_ratio(latest_goal_mask_heightmap, latest_depth_heightmap)
                shared_state['decreased_occupy_ratio'] = pre_occupy_ratio - occupy_ratio
                print('occupy ratio decrease:', shared_state['decreased_occupy_ratio'])
        
        print('episode step %d (max five pushes per episode)' % push_step)
        shared_state['push_step'] += 1
    
    return push_result


def execute_grasp_action(action_position, rotation_angle):
    """Execute grasp action and update state."""
    grasp_result, color_image, depth_image, color_height_map, depth_height_map, grasped_object_ind, gripper_success_open = \
        robot_agent.grasp(action_position, rotation_angle, workspace_bounds)
    
    print('Grasp result: %r' % grasp_result)
    
    if grasp_result:
        print('Grasp object: %d' % grasped_object_ind)
        model_trainer.grasp_obj_log.append([grasped_object_ind])
        data_logger.write_to_log('grasp-obj', model_trainer.grasp_obj_log)
    else:
        model_trainer.grasp_obj_log.append([-1])
        data_logger.write_to_log('grasp-obj', model_trainer.grasp_obj_log)
    
    shared_state['episode'] += 1
    
    if training_stage == 'push_only':
        print('episode step %d (five pushes make one episode)' % shared_state['push_step'])
        shared_state['push_step'] += 1
        shared_state['new_episode_flag'] = 1
    
    if grasp_result:
        if grasped_object_ind == shared_state['goal_obj_idx']:
            shared_state['goal_catched'] = 1
            print('Goal object captured!')
            shared_state['new_episode_flag'] = 1
            if is_testing:
                shared_state['restart_scene'] = robot_agent.num_obj / 2
        else:
            shared_state['goal_catched'] = 0.5
            print('Captured a non-goal object!')
            if not is_testing:
                shared_state['new_episode_flag'] = 1
    else:
        shared_state['goal_catched'] = 0
    
    if not gripper_success_open:
        print('gripper failed to open, restarting simulation')
        print('bus')
        robot_agent.restart_sim()
        robot_agent.add_objects()
    
    return grasp_result


def process_action_loop():
    """Main action processing loop running in separate thread."""
    while True:
        if shared_state['executing_action']:
            if check_object_out_of_bounds(shared_state['goal_obj_idx'], workspace_bounds, heightmap_pixel_size):
                shared_state['new_episode_flag'] = 1
                print("goal object has been pushed out of the workspace")
                robot_agent.restart_sim()
                robot_agent.add_objects()
                if is_testing:
                    model_trainer.model.load_state_dict(torch.load(snapshot_file))
                print('object is outside the view!')
                print('trial count: %d' % len(model_trainer.clearance_log))
                continue
            
            grasp_pred = shared_state['grasp_predictions']
            push_pred = shared_state['push_predictions']
            current_color_heightmap = shared_state.get('current_color_heightmap')
            current_valid_depth_heightmap = shared_state.get('current_valid_depth_heightmap')
            current_depth_heightmap = shared_state.get('current_depth_heightmap')
            current_goal_mask_heightmap = shared_state.get('current_goal_mask_heightmap')
            
            if current_color_heightmap is None:
                shared_state['executing_action'] = False
                continue
            
            masked_grasp_pred, masked_push_pred = apply_masks_to_predictions(
                grasp_pred, push_pred, current_color_heightmap, shared_state['current_goal_idx'])
            
            max_push_confidence = np.max(masked_push_pred)
            max_grasp_confidence = np.max(masked_grasp_pred)
            shared_state['grasp_reward'] = max_grasp_confidence
            print('Affordance scores — push: %f, grasp: %f' % (max_push_confidence, max_grasp_confidence))
            
            action_type = select_action_type(training_stage, max_push_confidence, max_grasp_confidence,
                                             shared_state['push_step'], shared_state['border_occupy_ratio'])
            shared_state['primitive_action'] = action_type
            
            if action_type == 'push':
                shared_state['best_pix_ind'] = np.unravel_index(np.argmax(masked_push_pred), masked_push_pred.shape)
                predicted_value = max_push_confidence
            elif action_type == 'grasp':
                shared_state['best_pix_ind'] = np.unravel_index(np.argmax(masked_grasp_pred), masked_grasp_pred.shape)
                predicted_value = max_grasp_confidence
            

            model_trainer.predicted_value_log.append([predicted_value])
            data_logger.write_to_log('predicted-value', model_trainer.predicted_value_log)

            print('Chosen %s at (%d, %d, %d)' % (action_type, shared_state['best_pix_ind'][0],
                                                   shared_state['best_pix_ind'][1], shared_state['best_pix_ind'][2]))
            
            action_position, rotation_angle = compute_action_position(
                shared_state['best_pix_ind'], current_valid_depth_heightmap, workspace_bounds,
                heightmap_pixel_size, action_type)
            
            pre_push_predictions = None
            pre_occupy_ratio = None
            if action_type == 'push':
                if training_stage == 'push_only':
                    pre_push_predictions, pre_occupy_ratio = prepare_push_only_metrics(
                        max_grasp_confidence, shared_state['push_step'], current_color_heightmap,
                        current_depth_heightmap, masked_grasp_pred, current_goal_mask_heightmap)
            
            if action_type == 'push':
                model_trainer.executed_action_log.append([0, shared_state['best_pix_ind'][0],
                                                          shared_state['best_pix_ind'][1],
                                                          shared_state['best_pix_ind'][2]])
            elif action_type == 'grasp':
                model_trainer.executed_action_log.append([1, shared_state['best_pix_ind'][0],
                                                          shared_state['best_pix_ind'][1],
                                                          shared_state['best_pix_ind'][2]])
            data_logger.write_to_log('executed-action', model_trainer.executed_action_log)
            
            if save_visualizations:
                push_pred_vis = model_trainer.get_prediction_vis(masked_push_pred, current_color_heightmap, shared_state['best_pix_ind'])
                data_logger.save_visualizations(model_trainer.iteration, push_pred_vis, 'push')
                grasp_pred_vis = model_trainer.get_prediction_vis(masked_grasp_pred, current_color_heightmap, shared_state['best_pix_ind'])
                data_logger.save_visualizations(model_trainer.iteration, grasp_pred_vis, 'grasp')
                push_direction_vis = model_trainer.get_best_push_direction_vis(shared_state['best_pix_ind'], current_color_heightmap)
                data_logger.save_visualizations(model_trainer.iteration, push_direction_vis, 'best_push_direction')
            
            shared_state['push_success'] = False
            shared_state['grasp_success'] = False
            
            if action_type == 'push':
                shared_state['push_success'] = execute_push_action(action_position, rotation_angle,
                                                                    shared_state['push_step'], max_grasp_confidence,
                                                                    pre_push_predictions, pre_occupy_ratio)
            elif action_type == 'grasp':
                shared_state['grasp_success'] = execute_grasp_action(action_position, rotation_angle)
            
            shared_state['executing_action'] = False
        
        time.sleep(0.01)


action_processing_thread = threading.Thread(target=process_action_loop)
action_processing_thread.daemon = True
action_processing_thread.start()
should_exit = False


def determine_current_goal(goal_mask_heightmap, goal_object):
    """Determine current goal object based on mask visibility."""
    white_pixel_count = np.sum(goal_mask_heightmap == 255)
    print('goal mask pixels in heightmap: %d' % white_pixel_count)
    
    if white_pixel_count <= 1 and white_pixel_count >= 0:
        obj_positions = np.asarray(robot_agent.get_obj_positions())
        obj_positions = obj_positions[:, 2]
        current_goal_idx = np.argmax(obj_positions)
        shared_state['current_goal_idx'] = current_goal_idx
        print('Mask too small — switch current goal to: %d' % shared_state['current_goal_idx'])
    else:
        shared_state['current_goal_idx'] = goal_object
        print('Mask sufficient — current goal remains: %d' % shared_state['current_goal_idx'])


def check_simulation_reset_conditions(valid_depth_heightmap, stagnation_counter):
    """Check if simulation needs to be reset."""
    object_count_map = np.zeros(valid_depth_heightmap.shape)
    object_count_map[valid_depth_heightmap > 0.02] = 1
    empty_threshold = 300
    if is_testing:
        empty_threshold = 10
    
    needs_restart = False
    if np.sum(object_count_map) < empty_threshold:
        print('Too few objects visible (%d) — repositioning.' % np.sum(object_count_map))
        needs_restart = True
    if stagnation_counter[0] + stagnation_counter[1] > 10:
        print('Excessive no-change count (%d) — repositioning.' % (stagnation_counter[0] + stagnation_counter[1]))
        needs_restart = True
    
    return needs_restart


def reset_simulation_environment():
    """Reset simulation and reload model if needed."""
    stagnation_counter = [0, 0]
    robot_agent.restart_sim()
    robot_agent.add_objects()
    if is_testing:
        model_trainer.model.load_state_dict(torch.load(snapshot_file))
    model_trainer.clearance_log.append([model_trainer.iteration])
    data_logger.write_to_log('clearance', model_trainer.clearance_log)
    print('Repositioning objects...')
    print('trial count: %d' % len(model_trainer.clearance_log))
    return stagnation_counter


def handle_episode_restart():
    """Handle episode restart logic."""
    shared_state['push_step'] = 0
    shared_state['new_episode_flag'] = 0
    
    print('starting episode %d' % shared_state['episode'])
    
    if shared_state['restart_scene'] == robot_agent.num_obj / 2:
        shared_state['restart_scene'] = 0
        stagnation_counter = [0, 0]
        robot_agent.restart_sim()
        robot_agent.add_objects()
        if is_testing:
            if use_cpu_only:
                device = torch.device("cpu")
                model_trainer.model.load_state_dict(torch.load(snapshot_file, map_location=device))
            else:
                model_trainer.model.load_state_dict(torch.load(snapshot_file))
    
    model_trainer.clearance_log.append([model_trainer.iteration])
    data_logger.write_to_log('clearance', model_trainer.clearance_log)
    print('trial count: %d' % len(model_trainer.clearance_log))
    print('begin a new episode')


def compute_network_predictions(color_heightmap, valid_depth_heightmap):
    """Compute network predictions based on stage and exploration settings."""
    if training_stage == 'grasp_only' and grasp_explore:
        should_explore = np.random.uniform() < grasp_exploration_probability
        
        if should_explore:
            print('Policy: explore (p=%f)' % grasp_exploration_probability)
            push_pred, grasp_pred, state_feat = model_trainer.forward(
                color_heightmap, valid_depth_heightmap, is_volatile=True, grasp_explore_actions=True)
            
            obj_contour = robot_agent.obj_contour(shared_state['current_goal_idx'])
            mask = robot_agent.mask(color_heightmap, shared_state['current_goal_idx'])
            mask = np.float32(mask)
            
            obj_grasp_prediction = np.multiply(grasp_pred, mask)
            grasp_pred = obj_grasp_prediction / 255
        else:
            print('Policy: exploit (p=%f)' % grasp_exploration_probability)
            push_pred, grasp_pred, state_feat = model_trainer.goal_forward(
                color_heightmap, valid_depth_heightmap, is_volatile=True)
    else:
        if not grasp_goal_conditioned:
            push_pred, grasp_pred, state_feat = model_trainer.forward(
                color_heightmap, valid_depth_heightmap, is_volatile=True)
        else:
            push_pred, grasp_pred, state_feat = model_trainer.goal_forward(
                color_heightmap, valid_depth_heightmap, is_volatile=True)
    
    return push_pred, grasp_pred


def detect_state_changes(prev_goal_mask_heightmap, prev_depth_heightmap, current_depth_heightmap):
    """Detect changes in goal object state."""
    prev_mask_hull = binary_dilation(convex_hull_image(prev_goal_mask_heightmap), iterations=5)
    depth_difference = prev_mask_hull * (prev_depth_heightmap - current_depth_heightmap)
    change_threshold = 100
    change_magnitude = utils.get_change_value(depth_difference)
    change_detected = change_magnitude > change_threshold
    print('Goal state change: %r (score: %d)' % (change_detected, change_magnitude))
    return change_detected, change_magnitude


def perform_experience_replay(prev_primitive_action, prev_reward_value):
    """Perform experience replay sampling and training."""
    sample_primitive_action = prev_primitive_action
    if grasp_goal_conditioned:
        sample_goal_obj_idx = shared_state['goal_obj_idx']
        print('sample_goal_obj_idx', sample_goal_obj_idx)
    
    if sample_primitive_action == 'push':
        sample_primitive_action_id = 0
        sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
    elif sample_primitive_action == 'grasp':
        sample_primitive_action_id = 1
        sample_reward_value = 0 if prev_reward_value == 1 else 1
    
    if not grasp_goal_conditioned or sample_primitive_action == 'push':
        sample_ind = np.argwhere(np.logical_and(
            np.asarray(model_trainer.reward_value_log)[0:model_trainer.iteration, 0] == sample_reward_value,
            np.asarray(model_trainer.executed_action_log)[0:model_trainer.iteration, 0] == sample_primitive_action_id))
    else:
        temp_idx1 = np.logical_and(
            np.asarray(model_trainer.reward_value_log)[0:model_trainer.iteration, 0] == sample_reward_value,
            np.asarray(model_trainer.grasp_obj_log)[0:model_trainer.iteration, 0] == sample_goal_obj_idx)
        temp_idx2 = np.logical_and(
            np.asarray(model_trainer.executed_action_log)[0:model_trainer.iteration, 0] == sample_primitive_action_id, temp_idx1)
        temp_idx3 = np.logical_and(
            np.asarray(model_trainer.current_grasp_obj_log)[0:model_trainer.iteration] == shared_state['current_goal_idx'], temp_idx2)
        sample_ind = np.argwhere(temp_idx3)
    
    if sample_ind.size > 0:
        sample_surprise_values = np.abs(np.asarray(model_trainer.predicted_value_log)[sample_ind[:, 0]] -
                                        np.asarray(model_trainer.label_value_log)[sample_ind[:, 0]])
        sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
        sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
        pow_law_exp = 2
        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
        sample_iteration = sorted_sample_ind[rand_sample_ind]
        print('Experience replay — sample %d (surprise: %f)' %
              (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
        
        sample_color_heightmap = cv2.imread(os.path.join(data_logger.color_heightmaps_directory,
                                                         '%06d.0.color.png' % sample_iteration))
        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
        sample_depth_heightmap = cv2.imread(os.path.join(data_logger.depth_heightmaps_directory,
                                                          '%06d.0.depth.png' % sample_iteration), -1)
        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000
        
        with torch.no_grad():
            if not grasp_goal_conditioned:
                sample_push_predictions, sample_grasp_predictions, sample_state_feat = model_trainer.forward(
                    sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
            else:
                sample_push_predictions, sample_grasp_predictions, sample_state_feat = model_trainer.goal_forward(
                    sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
        
        sample_best_pix_ind = (np.asarray(model_trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
        model_trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action,
                               sample_best_pix_ind, model_trainer.label_value_log[sample_iteration])
        
        if sample_primitive_action == 'push':
            model_trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
        elif sample_primitive_action == 'grasp':
            model_trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
    else:
        print('Insufficient prior samples — skip experience replay.')


# Main training/testing loop
while model_trainer.iteration < max_iterations:
    print('\n%s step: %d' % ('Testing' if is_testing else 'Training', model_trainer.iteration))
    iteration_start_time = time.time()
    
    color_img, depth_img = robot_agent.get_camera_data()
    depth_img = depth_img * robot_agent.cam_depth_scale
    color_heightmap, depth_heightmap = utils.get_heightmap(
        color_img, depth_img, robot_agent.cam_intrinsics, robot_agent.cam_pose,
        workspace_bounds, heightmap_pixel_size)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
    
    if not robot_agent.check_sim():
        print('Simulation became unstable — resetting environment.')
        print('bugs')
        stagnation_counter = [0, 0]
        robot_agent.restart_sim()
        robot_agent.add_objects()
    
    obj_contour = robot_agent.obj_contour(shared_state['goal_obj_idx'])
    utils.get_goal_coordinates(obj_contour, workspace_bounds, heightmap_pixel_size)
    
    goal_mask_heightmap = robot_agent.mask(color_heightmap, target_object_idx)
    goal_mask_heightmap = np.float32(goal_mask_heightmap)
    
    determine_current_goal(goal_mask_heightmap, target_object_idx)
    
    model_trainer.current_grasp_obj_log.append(shared_state['current_goal_idx'])
    data_logger.write_to_log('current-goal', model_trainer.current_grasp_obj_log)
    
    current_goal_mask_heightmap = robot_agent.mask(color_heightmap, shared_state['current_goal_idx'])
    current_goal_mask_heightmap = np.float32(current_goal_mask_heightmap)
    
    shared_state['border_occupy_ratio'] = utils.get_occupy_ratio(goal_mask_heightmap, depth_heightmap)
    
    data_logger.save_images(model_trainer.iteration, color_img, depth_img, '0')
    data_logger.save_heightmaps(model_trainer.iteration, color_heightmap, valid_depth_heightmap, '0')
    data_logger.save_visualizations(model_trainer.iteration, current_goal_mask_heightmap, 'mask')
    
    if check_simulation_reset_conditions(valid_depth_heightmap, stagnation_counter):
        stagnation_counter = reset_simulation_environment()
        if is_testing and len(model_trainer.clearance_log) >= max_test_trials:
            should_exit = True
        continue
    
    if (shared_state['push_step'] == max_push_episode_length + 1 or
            shared_state['new_episode_flag'] == 1 or
            shared_state['restart_scene'] == robot_agent.num_obj / 2):
        handle_episode_restart()
        if is_testing and len(model_trainer.clearance_log) >= max_test_trials:
            should_exit = True
        continue
    
    model_trainer.push_step_log.append([shared_state['push_step']])
    data_logger.write_to_log('push-step', model_trainer.push_step_log)
    
    if not should_exit:
        push_predictions, grasp_predictions = compute_network_predictions(color_heightmap, valid_depth_heightmap)
        
        shared_state['push_predictions'] = push_predictions
        shared_state['grasp_predictions'] = grasp_predictions
        shared_state['current_color_heightmap'] = color_heightmap
        shared_state['current_valid_depth_heightmap'] = valid_depth_heightmap
        shared_state['current_depth_heightmap'] = depth_heightmap
        shared_state['current_goal_mask_heightmap'] = current_goal_mask_heightmap
        
        shared_state['executing_action'] = True
    
    if 'prev_color_img' in locals():
        change_detected, change_value = detect_state_changes(
            prev_goal_mask_heightmap, prev_depth_heightmap, depth_heightmap)
        
        if change_detected:
            if prev_primitive_action == 'push':
                stagnation_counter[0] = 0
            elif prev_primitive_action == 'grasp':
                stagnation_counter[1] = 0
        else:
            if prev_primitive_action == 'push':
                stagnation_counter[0] += 1
            elif prev_primitive_action == 'grasp':
                stagnation_counter[1] += 1
        
        if not grasp_goal_conditioned:
            label_value, prev_reward_value = model_trainer.get_label_value(
                prev_primitive_action, prev_grasp_success, prev_grasp_reward,
                prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap)
        else:
            label_value, prev_reward_value = model_trainer.get_label_value(
                prev_primitive_action, prev_grasp_success, prev_grasp_reward,
                prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap,
                shared_state['goal_catched'], shared_state['decreased_occupy_ratio'])
        
        model_trainer.label_value_log.append([label_value])
        data_logger.write_to_log('label-value', model_trainer.label_value_log)
        model_trainer.reward_value_log.append([prev_reward_value])
        data_logger.write_to_log('reward-value', model_trainer.reward_value_log)
        
        model_trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action,
                               prev_best_pix_ind, label_value)
        
        if not is_testing:
            grasp_exploration_probability = max(0.8 * np.power(0.998, model_trainer.iteration), 0.1) if decay_exploration_rate else 0.8
        
        if use_experience_replay and not is_testing:
            perform_experience_replay(prev_primitive_action, prev_reward_value)
        
        if not is_testing:
            data_logger.save_backup_model(model_trainer.model, training_stage)
            if model_trainer.iteration % 100 == 0:
                data_logger.save_model(model_trainer.iteration, model_trainer.model, training_stage)
                if model_trainer.use_cuda:
                    model_trainer.model = model_trainer.model.cuda()
    
    while shared_state['executing_action']:
        time.sleep(0.01)
    
    if should_exit:
        break
    
    prev_color_img = color_img.copy()
    prev_depth_img = depth_img.copy()
    prev_color_heightmap = color_heightmap.copy()
    prev_depth_heightmap = depth_heightmap.copy()
    prev_valid_depth_heightmap = valid_depth_heightmap.copy()
    prev_push_success = shared_state['push_success']
    prev_grasp_success = shared_state['grasp_success']
    prev_primitive_action = shared_state['primitive_action']
    prev_push_predictions = push_predictions.copy()
    prev_grasp_predictions = grasp_predictions.copy()
    prev_best_pix_ind = shared_state['best_pix_ind']
    prev_goal_mask_heightmap = goal_mask_heightmap.copy()
    prev_grasp_reward = shared_state['grasp_reward']
    if training_stage == 'push_only':
        prev_improved_grasp_reward = shared_state['improved_grasp_reward']
    else:
        prev_improved_grasp_reward = 0.0
    
    model_trainer.iteration += 1
    
    iteration_end_time = time.time()
    print('Step time (s): %f' % (iteration_end_time - iteration_start_time))
