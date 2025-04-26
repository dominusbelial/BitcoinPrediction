import csv
import datetime
from pathlib import Path

def fix_bitcoin_csv(input_file, output_file):
    """
    Fix the Bitcoin price CSV file by completing the datetime column
    based on Unix timestamps.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to write the fixed CSV file
    """
    fixed_rows = []
    
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read the header row first
        header = next(reader, None)
        if header:
            fixed_rows.append(header)
        
        for row in reader:
            # Check if the datetime column is missing or incomplete
            if len(row) < 7 or not row[6].strip():
                # Calculate datetime from Unix timestamp
                unix_timestamp = float(row[0])
                datetime_str = datetime.datetime.fromtimestamp(
                    unix_timestamp, tz=datetime.timezone.utc
                ).strftime('%Y-%m-%d %H:%M:%S+00:00')
                
                # Create a new row with the calculated datetime
                new_row = row[:6] + [datetime_str]
                fixed_rows.append(new_row)
            else:
                # Row is already complete, keep as is
                fixed_rows.append(row)
    
    # Write the fixed data to the output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(fixed_rows)
    
    print(f"Fixed {len(fixed_rows)} rows and saved to {output_file}")

def main():
    input_file = 'data/btcusd_1-min_data.csv' #input("Enter the path to the Bitcoin CSV file: ")
    output_file = 'data/btcusd_1-min_data_fixed.csv' #input("Enter the path for the fixed CSV file: ")
    
    # Use default output filename if none provided
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    fix_bitcoin_csv(input_file, output_file)

if __name__ == "__main__":
    main()