import numpy as np
import argparse


parser = argparse.ArgumentParser(description='file name')
parser.add_argument('--output', dest='output', action='store', default='test')
args = parser.parse_args()
output_file_name = args.output

# fill in the output file name
with open(output_file_name) as f:
    lines = f.readlines()

graspPush = []
list_of_outcomes = []
pushes = []
graspsPushes = []


for line in lines:
    if 'Grasp successful:' in line:
        graspPush.append(line)
    if 'Push successful:' in line:
        graspPush.append(line)
    if 'Restart simluation!' in line:
        graspPush.append(line)
    if 'bugs' in line:
        graspPush.append(line)
    if 'goal object is pushed completely out of scene' in line:
        graspPush.append(line)
    if 'Too many no change counts' in line:
        graspPush.append(line)
    if 'Goal object catched!' in line:
        graspPush.append(line)




temp_cnt = 0
for line in graspPush:
    words = line.split()
    list_of_outcomes.append(words)
    if words[0] == "Restart":
        temp_cnt += 1
        if(temp_cnt > 2):
            graspsPushes.append(0)
    if words[0] == "Grasp" and words[-1] == "True":
        graspsPushes.append(1)
    if words[0] == "Grasp" and words[-1] == "False":
        graspsPushes.append(2)
    if words[0] == "Push" and words[-1] == "True":
        graspsPushes.append(3)
    if words[0] == "Push" and words[-1] == "False":
        graspsPushes.append(4)
    if words[-1] == "catched!":
        graspsPushes.append(5)
    if words[-1] == "scene":
        graspsPushes.append(6)
    if words[0] == "Too" and words[1] == "many":
        graspsPushes.append(7)
    if words[0] == "bugs":
        graspsPushes.append(8)

max_attempts = 5

#print(graspsPushes)

test_trial_idx = []
for i in range(len(graspsPushes)):
    if(graspsPushes[i] == 0):
        test_trial_idx.append(i)

test_trail_number = len(test_trial_idx) - 1


GraspsuccessCountTotal = []
motion_numberTotal = []
bugs_idx = []
out_scene_idx =[]
unchange_idx = []
success_idx = []

for i in range(test_trail_number):
    push_attempt = 0
    
    #print(graspsPushes[test_trial_idx[i]:test_trial_idx[i+1]])
    if(graspsPushes[test_trial_idx[i+1] - 1] == 5):
        success_idx.append(i)
        for j in range(test_trial_idx[i],test_trial_idx[i+1]):
            if(graspsPushes[j] == 1):#grasp true
                GraspsuccessCountTotal.append(1)
            if(graspsPushes[j] == 2):#grasp false
                GraspsuccessCountTotal.append(0)
            if(graspsPushes[j] == 3 or graspsPushes[j] == 4):
                push_attempt += 1
        motion_numberTotal.append(push_attempt)
    elif(graspsPushes[test_trial_idx[i+1] - 1] == 6):
        out_scene_idx.append(i)
    elif(graspsPushes[test_trial_idx[i+1]- 1] == 7):
        unchange_idx.append(i)
    elif(graspsPushes[test_trial_idx[i+1] - 1] == 8):
        bugs_idx.append(i)

#print(bugs_idx)
#print(out_scene_idx)
#print(unchange_idx)
#print(success_idx)

bugs_num = len(bugs_idx)
out_scene_num = len(out_scene_idx)
unchang_num = len(unchange_idx)
success_num = len(success_idx)
sum_number = out_scene_num+unchang_num+success_num
print([bugs_num,out_scene_num,unchang_num,success_num, sum_number])
print("final_completion %", success_num/sum_number * 100)
final_GraspSuccessRate = (sum(GraspsuccessCountTotal)/len(GraspsuccessCountTotal))*100
print("final_GraspSuccessRate %", final_GraspSuccessRate)
motion_number =  (sum(motion_numberTotal)/len(motion_numberTotal))
print("motion_number", motion_number)