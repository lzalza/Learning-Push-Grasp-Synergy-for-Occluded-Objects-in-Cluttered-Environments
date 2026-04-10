python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_1.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 1 --max_test_trials 100 >  "$logs1"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_2.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 1 --max_test_trials 100 >  "$logs2"


python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_3.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --max_test_trials 100 >  "$logs3"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_4.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 2 --max_test_trials 100 >  "$logs4"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_5.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 4 --max_test_trials 100 >  "$logs5"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_6.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 5 --max_test_trials 100 >  "$logs6"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_7.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --max_test_trials 100 >  "$logs7"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_8.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 0 --max_test_trials 100 >  "$logs8"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_9.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 2 --max_test_trials 100 >  "$logs9"

python main.py --stage push_grasp --num_obj 20 --experience_replay --explore_rate_decay --is_testing --test_preset_cases --test_preset_file 'simulation/test/obj_20_10.txt' \
                                --load_snapshot --snapshot_file "$X5"                                --save_visualizations --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 3 --max_test_trials 100 >  "$logs10"
