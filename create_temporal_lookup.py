# create_temporal_lookup_fixed.py
# Handle the ID format mismatch and missing years properly

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

def create_temporal_lookup():
    """Create temporal variables handling ID format issues"""
    
    print("Loading data...")
    
    # Load paper IDs mapping
    with open('data/sim_bm25/paper_ids.json', 'r') as f:
        paper_ids_list = json.load(f)  # ['w0002', 'w0003', ...]
    
    # Create mapping from index to paper_id string
    # The index in this list corresponds to the integer ID used in similarity data
    idx_to_paperid = {i: pid for i, pid in enumerate(paper_ids_list)}
    
    # Load years from meta_sets
    with open('subset/meta_sets.pkl', 'rb') as f:
        meta_sets = pickle.load(f)  # {3: {...}, 6: {...}, ...}
    
    # Create year lookup using integer IDs
    paper_years = {}
    for int_id, meta in meta_sets.items():
        if 'year' in meta:
            paper_years[int_id] = meta['year']
    
    print(f"Found years for {len(paper_years)} papers out of {len(paper_ids_list)} total")
    print(f"Year range: {min(paper_years.values())} - {max(paper_years.values())}")
    
    # Process one method to get all unique pairs
    temporal_data = []
    processed_pairs = set()
    
    # Just use BM25 to get all pairs (they should be the same across methods)
    method = 'bm25'
    print(f"\nProcessing {method} pairs to extract temporal data...")
    
    for chunk_idx in range(7):
        chunk_file = f'data/paper_similarity_chunks/paper_similarity_dataset_{method}_blk{chunk_idx:05d}.csv'
        
        if not Path(chunk_file).exists():
            print(f"  Chunk {chunk_idx} not found, skipping...")
            continue
        
        print(f"  Processing chunk {chunk_idx}...")
        chunk_df = pd.read_csv(chunk_file, chunksize=100000)  # Read in chunks for memory efficiency
        
        for batch_idx, batch in enumerate(chunk_df):
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}, processed {len(temporal_data):,} pairs so far...")
            
            for _, row in batch.iterrows():
                paper_i = int(row['paper_i'])
                paper_j = int(row['paper_j'])
                
                # Skip if we've already processed this pair
                pair_key = (paper_i, paper_j)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Get years
                year_i = paper_years.get(paper_i, None)
                year_j = paper_years.get(paper_j, None)
                
                # Skip if either year is missing
                if year_i is None or year_j is None:
                    continue
                
                # Calculate temporal variables
                year_diff = year_j - year_i
                year_diff_abs = abs(year_diff)
                
                temporal_data.append({
                    'paper_i': paper_i,
                    'paper_j': paper_j,
                    'year_i': year_i,
                    'year_j': year_j,
                    'year_diff': year_diff,
                    'year_diff_abs': year_diff_abs,
                    'same_year': int(year_diff_abs == 0),
                    'within_1yr': int(year_diff_abs <= 1),
                    'within_2yrs': int(year_diff_abs <= 2),
                    'within_3yrs': int(year_diff_abs <= 3),
                    'within_5yrs_new': int(year_diff_abs <= 5),
                    'within_10yrs': int(year_diff_abs <= 10),
                    'post_genai': int(year_i >= 2021.5 and year_j >= 2021.5),
                    'post_gpt3': int(year_i >= 2020.5 and year_j >= 2020.5),
                    'post_chatgpt': int(year_i >= 2022.9 and year_j >= 2022.9)
                })
    
    print(f"\nCreated temporal data for {len(temporal_data):,} pairs")
    
    if len(temporal_data) == 0:
        print("ERROR: No temporal data created!")
        return None
    
    # Convert to DataFrame
    temporal_df = pd.DataFrame(temporal_data)
    
    # Save as CSV first
    temporal_df.to_csv('data/temporal_variables_lookup.csv', index=False)
    print("Saved CSV version")
    
    # For Stata, ensure proper data types
    # Paper IDs should be int64 to match your Stata files
    temporal_df['paper_i'] = temporal_df['paper_i'].astype(np.int64)
    temporal_df['paper_j'] = temporal_df['paper_j'].astype(np.int64)
    
    # Save as Stata file
    temporal_df.to_stata('data/temporal_variables_lookup.dta', write_index=False)
    print("Saved Stata version")
    
    # Print summary statistics
    print("\n=== Summary of Temporal Variables ===")
    print(f"Total pairs with year data: {len(temporal_df):,}")
    print(f"Percentage of total pairs: {100 * len(temporal_df) / (33411 * 33411):.1f}%")
    print(f"\nTemporal distribution:")
    print(f"  Same year: {temporal_df['same_year'].sum():,} ({100*temporal_df['same_year'].mean():.1f}%)")
    print(f"  Within 2 years: {temporal_df['within_2yrs'].sum():,} ({100*temporal_df['within_2yrs'].mean():.1f}%)")
    print(f"  Within 5 years: {temporal_df['within_5yrs_new'].sum():,} ({100*temporal_df['within_5yrs_new'].mean():.1f}%)")
    print(f"\nGenAI era analysis:")
    print(f"  Post-GenAI (July 2021): {temporal_df['post_genai'].sum():,} ({100*temporal_df['post_genai'].mean():.1f}%)")
    print(f"  Post-GPT3 (July 2020): {temporal_df['post_gpt3'].sum():,} ({100*temporal_df['post_gpt3'].mean():.1f}%)")
    print(f"  Post-ChatGPT (Sept 2022): {temporal_df['post_chatgpt'].sum():,} ({100*temporal_df['post_chatgpt'].mean():.1f}%)")
    
    # Year distribution
    print(f"\nYear range in data: {temporal_df['year_i'].min()} - {temporal_df['year_i'].max()}")
    
    return temporal_df

if __name__ == "__main__":
    temporal_df = create_temporal_lookup()
    
    
df = temporal_df.head(2000)
