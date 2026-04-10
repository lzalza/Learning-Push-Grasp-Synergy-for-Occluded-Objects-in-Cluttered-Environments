# Learning-Push-Grasp-Synergy-for-Occluded-Objects-in-Cluttered-Environments

This repository contains the official implementation of the code for the paper: "Learning Push-Grasp Synergy for Occluded Objects in Cluttered Environments". 

## Simulation Environment

The experiments and simulations are conducted using **CoppeliaSim** (formerly V-REP). 

* **Software:** CoppeliaSim
* **Version:** V4.1.0

Please ensure you have downloaded and installed the correct version of CoppeliaSim (V4.1.0) to ensure compatibility. You can download it from the [official Coppelia Robotics website](https://www.coppeliarobotics.com/).

## Training

The training process is divided into three main steps. Please run the following commands in order:

**1. Training GraspNet**

Run the following command to start training the GraspNet module:
```bash
python main.py --stage grasp_only --num_obj 10 --goal_conditioned --goal_obj_idx 0 --experience_replay --explore_rate_decay --save_visualizations --logging_directory "$Y1"
python main.py --stage grasp_only --num_obj 10 --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --experience_replay --explore_rate_decay --save_visualizations --grasp_explore --load_explore_snapshot --explore_snapshot_file "$X1" --logging_directory "$Y2"
```

**2. Training PushNet**

Next, train the PushNet module using this command:

```bash
python main.py --stage push_only --num_obj 20 --grasp_reward_threshold 1.8 --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --experience_replay --explore_rate_decay --save_visualizations  --load_snapshot --snapshot_file "$X2" --logging_directory "$Y3"
```

**3. Alternating Training**

Finally, perform the alternating training for both networks:

```bash
python main.py --stage push_only --num_obj 20 --alternating_training --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --experience_replay --explore_rate_decay --save_visualizations  --load_snapshot --snapshot_file "$X3" --logging_directory "$Y4"
python main.py --stage push_only --num_obj 20 --grasp_reward_threshold 1.8 --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --experience_replay --explore_rate_decay --save_visualizations  --load_snapshot --snapshot_file "$X4" --logging_directory "$Y5"
```

## Testing

To evaluate the trained models, follow these steps:

**1. Run Testing**

Execute the testing script to evaluate the models in ten severely occluded simulation scenarios

```bash
sh run_test.sh
```

**2. Evaluate Results**

After the tests have finished running, evaluate the logs of the test using:

```bash
python evaluate.py --output "$logs"
```

## Acknowledgment

Parts of this code are based on and inspired by [Self-Supervised Learning for Joint Pushing and Grasping Policies in Highly Cluttered Environments](https://github.com/Kamalnl92/Self-Supervised-Learning-for-pushing-and-grasping).
