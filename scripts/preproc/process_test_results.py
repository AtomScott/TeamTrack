import os
import pandas as pd
import argparse
import tabulate

def process_results(results_dir, output_dir):
    """
    This function reads pedestrian_summary.txt files from the given directory,
    extracts metrics of interest, and saves them in a DataFrame.
    The DataFrame is then saved to a separate CSV file in the output directory for each dataset,
    and printed in markdown format.
    
    Args:
    results_dir (str): path to the directory containing the result directories
    output_dir (str): path to the directory where the output CSV files will be saved
    """

    # Check if results_dir exists
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} does not exist or is not a directory.")
        return

    # Prepare the dataframe
    headers = ['Methods', 'HOTA', 'DetA', 'AssA', 'MOTA', 'IDSW', 'IDF1', 'MOTP', 'CLR_Pr', 'CLR_Re', 'MT']

    # Walk through the directory tree
    for dataset_name in os.listdir(results_dir):
        dataset_dir = os.path.join(results_dir, dataset_name)
        if not os.path.isdir(dataset_dir): continue        
        # Initialize a new DataFrame for each dataset
        results_df = pd.DataFrame(columns=headers)

        for method_name in os.listdir(dataset_dir):
            method_dir = os.path.join(dataset_dir, method_name)
            if not os.path.isdir(method_dir): continue
            file = os.path.join(method_dir, 'pedestrian_summary.txt')
            
            # Read the file
            with open(file, 'r') as f:
                lines = f.readlines()
                
                # Extract the scores
                scores = lines[1].split()

                # Match the scores to the metrics in the headers
                scores_dict = dict(zip(lines[0].split(), scores))
                scores_to_add = [scores_dict.get(h, '') for h in headers[1:]]  # Exclude 'Methods'
                
                # Add the row to the dataframe
                results_df.loc[method_name] = [method_name] + scores_to_add

        # Save the DataFrame to a CSV file in the output directory
        output_file = os.path.join(output_dir, os.path.basename(dataset_dir), 'results.csv')
        results_df.to_csv(output_file, index=False)

        # Print the DataFrame in markdown format using tabulate
        print(f"### {os.path.basename(dataset_name)}\n")
        print(tabulate.tabulate(results_df, headers='keys', tablefmt='pipe', showindex=False))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tracking results and produce a summary table.")
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the directory containing the result directories.')
    parser.add_argument('--output_dir', type=str, help='Path to the directory where the output CSV files will be saved. Default is the same as results_dir.')
    args = parser.parse_args()

    # If output_dir is not specified, default to results_dir
    if args.output_dir is None:
        args.output_dir = args.results_dir

    process_results(args.results_dir, args.output_dir)
