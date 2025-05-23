import pandas as pd

def deaccumulate_rewards(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['Reward'] = df['Reward'].diff().fillna(df['Reward'])
    df.to_csv(output_csv, index=False)
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python deaccumulate_rewards.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    deaccumulate_rewards(input_csv, output_csv)