import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TeamTrack evaluation script')
    parser.add_argument('--input-dir', type=str, default='./', help='Path to input directory containing test data and parameter yaml file')
    parser.add_argument('--output-dir', type=str, default='./', help='Path to save output csv file')
    return parser.parse_args()

def main(args):
    # TODO: Load the model

    # TODO: Load the test data from args.input_dir

    # TODO: Load the parameters from the yaml file in args.input_dir

    # TODO: Evaluate the model on the test data using the loaded parameters

    # TODO: Save the results to a csv file in args.output_dir

if __name__ == "__main__":
    args = parse_args()
    main(args)
