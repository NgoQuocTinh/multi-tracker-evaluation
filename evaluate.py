"""
evaluate.py - Compare tracker outputs against ground truth using MOT metrics.
Uses fps_log.txt to get runtime and FPS.
"""

import os
import motmetrics as mm
import numpy as np
import pandas as pd

# CONFIG
GT_FILE = 'data/gt.txt'
OUTPUT_CSV = 'evaluation_results_1/tracker_comparison.csv'

TRACKERS = {
    'DeepSORT': 'results_1/results_deepsort.txt',
    'BOTSort': 'results_1/results_botsort.txt',
    'ByteTrack': 'results_1/results_bytetrack.txt',
    'SORT': 'results_1/results_sort.txt'
}

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

def load_tracking_results(file_path):
    """Load tracking results or ground truth from a CSV file into a DataFrame and compute derived coordinates."""
    data = pd.read_csv(file_path, header=None,
                       names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
    data['x2'] = data['x'] + data['w']
    data['y2'] = data['y'] + data['h']
    return data

def calculate_mot_metrics(gt, predictions):
    """Compute MOT metrics for a single tracker given the ground truth and predictions."""
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in sorted(gt['frame'].unique()):
        gt_boxes = gt[gt['frame'] == frame_id][['x', 'y', 'x2', 'y2']].values
        gt_ids = gt[gt['frame'] == frame_id]['id'].values

        pred_boxes = predictions[predictions['frame'] == frame_id][['x', 'y', 'x2', 'y2']].values
        pred_ids = predictions[predictions['frame'] == frame_id]['id'].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'motp', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'mostly_lost'],
        name='acc'
    )
    return summary

def read_fps_runtime(tracker_name, tracker_file):
    """Read runtime and FPS from fps_log.txt in the same folder as tracker results."""
    folder = os.path.dirname(tracker_file)
    fps_file = os.path.join(folder, "fps_log.txt")
    runtime_sec = None
    fps_value = None
    if os.path.exists(fps_file):
        with open(fps_file, "r") as f:
            for line in f:
                # Split line by '|'
                parts = [p.strip() for p in line.strip().split('|')]
                if len(parts) < 4:
                    continue
                name = parts[0].strip()
                # Compare exact match, ignore case
                if name.lower() == tracker_name.lower():
                    for part in parts[1:]:
                        if part.lower().startswith('time:'):
                            runtime_sec = float(part.split(':')[1].strip().replace('s',''))
                        elif part.lower().startswith('fps:'):
                            fps_value = float(part.split(':')[1].strip())
                    break
    return runtime_sec, fps_value


def main():
    print("Loading ground truth...")
    gt = load_tracking_results(GT_FILE)

    all_metrics = []

    for tracker_name, tracker_file in TRACKERS.items():
        print(f"\nProcessing tracker: {tracker_name}")
        predictions = load_tracking_results(tracker_file)
        mot_summary = calculate_mot_metrics(gt, predictions)

        runtime_sec, fps = read_fps_runtime(tracker_name, tracker_file)

        tracker_metrics = {
            'Tracker': tracker_name,
            'MOTA': mot_summary['mota']['acc'],
            'MOTP': mot_summary['motp']['acc'],
            'ID Switches': mot_summary['num_switches']['acc'],
            'Fragmentations': mot_summary['num_fragmentations']['acc'],
            'Mostly Tracked (%)': mot_summary['mostly_tracked']['acc'],
            'Mostly Lost (%)': mot_summary['mostly_lost']['acc'],
            'Average Track Length': predictions.groupby('id').size().mean(),
            'Estimated ID Switches': predictions.groupby('id')['frame']
                .apply(lambda x: np.sum(np.diff(sorted(x)) > 1)).sum(),
            'Runtime (s)': runtime_sec if runtime_sec else 'NA',
            'FPS': fps if fps else 'NA'
        }

        all_metrics.append(tracker_metrics)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(OUTPUT_CSV, index=False)

    print("\nComparative Tracker Evaluation Metrics:")
    print(comparison_df.to_string(index=False))

if __name__ == '__main__':
    main()
