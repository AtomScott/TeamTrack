import argparse
import os
import shutil
from glob import glob


def check_pairs(annotations, videos):
    # Extract the base names without extensions
    annotation_bases = set(os.path.splitext(os.path.basename(path))[0] for path in annotations)
    video_bases = set(os.path.splitext(os.path.basename(path))[0] for path in videos)

    # Find and print the files with no pairs
    annotations_without_pairs = annotation_bases.difference(video_bases)
    videos_without_pairs = video_bases.difference(annotation_bases)

    if annotations_without_pairs:
        print("Annotations with no corresponding video:")
        for name in annotations_without_pairs:
            print(name)

    if videos_without_pairs:
        print("Videos with no corresponding annotation:")
        for name in videos_without_pairs:
            print(name)
    return not bool(annotations_without_pairs or videos_without_pairs)


# Setup argument parser
parser = argparse.ArgumentParser(description="Split the TeamTrack dataset into train, val and test.")
parser.add_argument("--input-dir", default="./teamtrack", type=str, help="Path to the TeamTrack data")
parser.add_argument("--output-dir", default="./teamtrack_split", type=str, help="Path to save the split TeamTrack data")
parser.add_argument("--train-ratio", default=0.7, type=float, help="Ratio of data to use for training")
parser.add_argument("--val-ratio", default=0.15, type=float, help="Ratio of data to use for validation")
parser.add_argument("--test-ratio", default=0.15, type=float, help="Ratio of data to use for testing")
args = parser.parse_args()

dataset_names = [
    "Basketball_SideView",
    "Basketball_SideView2",
    "Basketball_TopView",
    "Handball_SideView",
    "Soccer_SideView",
    "Soccer_TopView",
]

for dataset_name in dataset_names:
    dataset_path = os.path.join(args.input_dir, dataset_name)
    output_path = os.path.join(args.output_dir, dataset_name)
    annotations = sorted(glob(os.path.join(dataset_path, "annotations", "*.csv")))
    videos = sorted(glob(os.path.join(dataset_path, "videos", "*.mp4")))
    print("=" * 80)
    print(f"Saving to {output_path}")
    print(f"Dataset: {dataset_name}")

    n_annotations = len(annotations)
    n_videos = len(videos)
    assert check_pairs(annotations, videos), "Pairs are missing"
    n_seqs = n_annotations

    # Calculate split indices
    train_idx = round(n_seqs * args.train_ratio)
    val_idx = train_idx + round(n_seqs * args.val_ratio)

    # Split annotations
    train_annotations = annotations[:train_idx]
    val_annotations = annotations[train_idx:val_idx]
    test_annotations = annotations[val_idx:]

    # Split videos
    train_videos = videos[:train_idx]
    val_videos = videos[train_idx:val_idx]
    test_videos = videos[val_idx:]

    # print stats
    print(f"Total sequences: {n_seqs}")
    print(f"Train annotations: {len(train_annotations)}, Train videos: {len(train_videos)}")
    print(f"Val annotations: {len(val_annotations)}, Val videos: {len(val_videos)}")
    print(f"Test annotations: {len(test_annotations)}, Test videos: {len(test_videos)}")

    # Create directories and move sequences
    # for subset_annotation, subset_videos, subset in zip([train_dirs, val_dirs, test_dirs], ["train", "val", "test"]):
    subsets = [
        (train_annotations, train_videos, "train"),
        (val_annotations, val_videos, "val"),
        (test_annotations, test_videos, "test"),
    ]

    for subset_annotations, subset_videos, subset in subsets:
        for annotation, video in zip(subset_annotations, subset_videos):
            annotation_name = os.path.basename(annotation)
            video_name = os.path.basename(video)
            assert annotation_name[:-3] == video_name[:-3], f"Annotation and video names do not match: {annotation_name} != {video_name}"

            annotation_path = os.path.join(output_path, subset, "annotations", annotation_name)
            video_path = os.path.join(output_path, subset, "videos", video_name)

            os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            shutil.copy(annotation, annotation_path)
            shutil.copy(video, video_path)

# python split_teamtrack.py --input-dir ./teamtrack --output-dir ./teamtrack_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
