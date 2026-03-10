import argparse
from typing import List, Tuple


# Event codes for clarity (preserving original numeric semantics)
# 0: trial separator (after >2 restarts)
# 1: grasp true, 2: grasp false, 3: push true, 4: push false
# 5: goal captured, 6: out of workspace, 7: excessive no-change, 8: bugs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse evaluation log and compute metrics')
    parser.add_argument('--output', dest='output', action='store', default='test')
    return parser.parse_args()


def read_lines(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")
    except OSError as exc:
        raise SystemExit(f"Could not read file {path}: {exc}")


def filter_relevant_lines(lines: List[str]) -> List[str]:
    relevant: List[str] = []
    for line in lines:
        if 'Grasp result:' in line:
            relevant.append(line)
        elif 'Push result:' in line:
            relevant.append(line)
        elif 'Restart simluation!' in line:
            relevant.append(line)
        elif 'bugs' in line:
            relevant.append(line)
        elif 'goal object has been pushed out of the workspace' in line:
            relevant.append(line)
        elif 'Excessive no-change count' in line:
            relevant.append(line)
        elif 'Goal object captured!' in line:
            relevant.append(line)
    return relevant


def map_lines_to_events(relevant_lines: List[str]) -> List[int]:
    events: List[int] = []
    restart_count = 0

    for line in relevant_lines:
        words = line.split()
        if not words:
            continue

        # Restart handling first to preserve side-effects
        if words[0] == 'Restart':
            restart_count += 1
            if restart_count > 2:
                events.append(0)
            continue

        # Outcome mappings
        if words[0] == 'Grasp' and words[-1] == 'True':
            events.append(1)
            continue
        if words[0] == 'Grasp' and words[-1] == 'False':
            events.append(2)
            continue
        if words[0] == 'Push' and words[-1] == 'True':
            events.append(3)
            continue
        if words[0] == 'Push' and words[-1] == 'False':
            events.append(4)
            continue
        if words[-1] == 'captured!':
            events.append(5)
            continue
        if words[-1] == 'workspace':
            events.append(6)
            continue
        if words[0] == 'Excessive':
            events.append(7)
            continue
        if words[0] == 'bugs':
            events.append(8)
            continue

    return events


def collect_trial_indices(events: List[int]) -> List[int]:
    indices: List[int] = []
    for idx, code in enumerate(events):
        if code == 0:
            indices.append(idx)
    return indices


def compute_metrics(events: List[int], trial_indices: List[int]) -> Tuple[int, int, int, int, int, float, float]:
    grasp_success_markers: List[int] = []
    motion_counts_per_success: List[int] = []
    bugs_trials: List[int] = []
    out_scene_trials: List[int] = []
    unchanged_trials: List[int] = []
    success_trials: List[int] = []

    num_trials = max(len(trial_indices) - 1, 0)
    for i in range(num_trials):
        push_attempts_in_trial = 0
        start = trial_indices[i]
        end = trial_indices[i + 1]

        if end <= 0:
            continue

        last_code = events[end - 1]
        if last_code == 5:
            success_trials.append(i)
            for j in range(start, end):
                if events[j] == 1:
                    grasp_success_markers.append(1)
                elif events[j] == 2:
                    grasp_success_markers.append(0)
                elif events[j] in (3, 4):
                    push_attempts_in_trial += 1
            motion_counts_per_success.append(push_attempts_in_trial)
        elif last_code == 6:
            out_scene_trials.append(i)
        elif last_code == 7:
            unchanged_trials.append(i)
        elif last_code == 8:
            bugs_trials.append(i)

    bugs_num = len(bugs_trials)
    out_scene_num = len(out_scene_trials)
    unchanged_num = len(unchanged_trials)
    success_num = len(success_trials)
    total_considered = out_scene_num + unchanged_num + success_num

    completion_percent = (success_num / total_considered * 100) if total_considered > 0 else 0.0
    grasp_success_percent = (
        (sum(grasp_success_markers) / len(grasp_success_markers) * 100)
        if len(grasp_success_markers) > 0 else 0.0
    )
    avg_motion_number = (
        (sum(motion_counts_per_success) / len(motion_counts_per_success))
        if len(motion_counts_per_success) > 0 else 0.0
    )

    return (
        bugs_num,
        out_scene_num,
        unchanged_num,
        success_num,
        total_considered,
        completion_percent,
        avg_motion_number,
    ), grasp_success_percent


def main() -> None:
    args = parse_args()
    lines = read_lines(args.output)
    relevant_lines = filter_relevant_lines(lines)
    events = map_lines_to_events(relevant_lines)
    trial_indices = collect_trial_indices(events)

    (bugs_num,
     out_scene_num,
     unchanged_num,
     success_num,
     total_considered,
     completion_percent,
     avg_motion_number), grasp_success_percent = compute_metrics(events, trial_indices)

    print([bugs_num, out_scene_num, unchanged_num, success_num, total_considered])
    print("final_completion %", completion_percent)
    print("final_GraspSuccessRate %", grasp_success_percent)
    print("motion_number", avg_motion_number)


if __name__ == '__main__':
    main()