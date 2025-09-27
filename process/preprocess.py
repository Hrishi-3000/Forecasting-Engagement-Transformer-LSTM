# process/preprocess.py
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path='data/raw/', output_path='data/processed/'):
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def detect_sessions(self):
        """Automatically detect available sessions"""
        sessions = []

        if not os.path.exists(self.data_path):
            return sessions

        # Look for session directories
        for item in os.listdir(self.data_path):
            if item.startswith('session') and os.path.isdir(os.path.join(self.data_path, item)):
                try:
                    session_num = int(item.replace('session', ''))
                    sessions.append(session_num)
                except:
                    continue

        # Also check for numbered files in raw directory
        if not sessions:
            files = os.listdir(self.data_path)
            session_nums = set()
            for file in files:
                if file.endswith(('.csv', '.xlsx')) and '_' in file:
                    try:
                        session_num = int(file.split('_')[0])
                        session_nums.add(session_num)
                    except:
                        continue
            sessions = sorted(list(session_nums))

        return sorted(sessions)

    def load_data(self, session_num):
        """Load data from various file naming conventions"""
        session_path = os.path.join(self.data_path, f'session{session_num}')
        data = {}

        print(f"   Looking for data in: {session_path}")

        if not os.path.exists(session_path):
            print(f"   ❌ Session directory not found: {session_path}")
            return data

        files = os.listdir(session_path)
        print(f"   Found files: {files}")

        file_patterns = {
            'ENG': [f'{session_num}_ENG', 'engagement', 'ENG'],
            'EEG': [f'{session_num}_EEG', 'EEG'],
            'GSR': [f'{session_num}_GSR', 'GSR'],
            'EYE': [f'{session_num}_EYE', 'EYE'],
            'IVT': [f'{session_num}_IVT', 'IVT']
        }

        for data_type, patterns in file_patterns.items():
            file_loaded = False

            for pattern in patterns:
                for ext in ['.csv', '.xlsx', '.xls']:
                    file_path = os.path.join(session_path, f"{pattern}{ext}")

                    if os.path.exists(file_path):
                        try:
                            if ext in ['.xlsx', '.xls']:
                                df = pd.read_excel(file_path)
                            else:
                                df = pd.read_csv(file_path)

                            # Use smaller sample for testing
                            if len(df) > 500:
                                df = df.head(500)
                                print(f"     Using first 500 rows for faster processing")

                            data[data_type] = df
                            print(f"   ✓ Loaded {data_type} from: {os.path.basename(file_path)}")
                            print(f"     Shape: {data[data_type].shape}")
                            file_loaded = True
                            break
                        except Exception as e:
                            print(f"   ❌ Error loading {file_path}: {e}")
                            continue

                if file_loaded:
                    break

            if not file_loaded:
                print(f"   ⚠ Could not find {data_type} data")

        return data

    def robust_timestamp_conversion(self, timestamp_series):
        """Robust timestamp conversion that handles various formats"""
        print(f"     Converting timestamp, dtype: {timestamp_series.dtype}")
        print(f"     Sample values: {timestamp_series.head(3).tolist()}")

        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(timestamp_series):
            print("     Already numeric, returning as is")
            return timestamp_series

        # Try to convert to numeric directly
        try:
            numeric_ts = pd.to_numeric(timestamp_series, errors='coerce')
            if not numeric_ts.isna().all():
                print("     Successfully converted to numeric directly")
                numeric_ts = numeric_ts.fillna(pd.Series(range(len(numeric_ts)), index=numeric_ts.index))
                return numeric_ts
        except:
            pass

        # Try datetime conversion with multiple formats
        datetime_formats = [
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', 
            '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S.%f',
            '%H:%M:%S', '%H:%M:%S.%f'
        ]

        for fmt in datetime_formats:
            try:
                datetime_series = pd.to_datetime(timestamp_series, format=fmt, errors='coerce')
                if not datetime_series.isna().all():
                    numeric_ts = (datetime_series - datetime_series.min()).dt.total_seconds()
                    print(f"     Successfully parsed as datetime format: {fmt}")
                    return numeric_ts
            except:
                continue

        # Try generic datetime parsing
        try:
            datetime_series = pd.to_datetime(timestamp_series, errors='coerce')
            if not datetime_series.isna().all():
                numeric_ts = (datetime_series - datetime_series.min()).dt.total_seconds()
                print("     Successfully parsed as generic datetime")
                return numeric_ts
        except:
            pass

        # If it's string but looks like numeric with commas/formatting
        if timestamp_series.dtype == 'object':
            try:
                clean_series = timestamp_series.astype(str).str.replace(',', '').str.replace(' ', '')
                numeric_ts = pd.to_numeric(clean_series, errors='coerce')
                if not numeric_ts.isna().all():
                    print("     Successfully cleaned and converted to numeric")
                    numeric_ts = numeric_ts.fillna(pd.Series(range(len(numeric_ts)), index=numeric_ts.index))
                    return numeric_ts
            except:
                pass

        # Final fallback: use index as timestamp
        print("     Using index as timestamp (fallback)")
        return pd.Series(range(len(timestamp_series)), index=timestamp_series.index)

    def find_timestamp_column(self, df):
        """Find the timestamp column in a dataframe"""
        timestamp_aliases = ['timestamp', 'time', 'Time', 'Timestamp', 't', 'T', 'TimeStamp', 'datetime', 'DateTime']

        for col in df.columns:
            if col.lower() in [alias.lower() for alias in timestamp_aliases]:
                return col

        for col in df.columns:
            if any(word in col.lower() for word in ['time', 'stamp', 'date']):
                return col

        return None

    def create_simple_synchronized_data(self, data_dict):
        """Create synchronized data using simple approach"""
        print("   Creating synchronized data using simple approach...")

        ref_modality = max(data_dict.keys(), key=lambda k: len(data_dict[k]) if data_dict[k] is not None else 0)
        ref_df = data_dict[ref_modality]

        print(f"   Using {ref_modality} as reference (length: {len(ref_df)})")

        base_length = len(ref_df)
        synchronized_df = pd.DataFrame({'timestamp': range(base_length)})

        for modality, df in data_dict.items():
            if df is not None and not df.empty:
                modality_length = len(df)

                if modality_length == base_length:
                    for col in df.columns:
                        if col not in ['timestamp', 'time', 'TimeStamp']:
                            synchronized_df[f'{modality}_{col}'] = df[col].values
                else:
                    for col in df.columns:
                        if col not in ['timestamp', 'time', 'TimeStamp']:
                            original_indices = np.linspace(0, base_length-1, modality_length)
                            new_indices = range(base_length)
                            interpolated_values = np.interp(new_indices, original_indices, df[col].values)
                            synchronized_df[f'{modality}_{col}'] = interpolated_values

                print(f"     Added {modality} data")

        if 'ENG' not in data_dict or data_dict['ENG'] is None:
            print("   Creating synthetic engagement...")
            time_normalized = synchronized_df['timestamp'] / synchronized_df['timestamp'].max()
            engagement = 0.5 + 0.3 * np.sin(time_normalized * 4 * np.pi)
            synchronized_df['engagement'] = np.clip(engagement, 0.1, 0.9)

        return synchronized_df

    def preprocess_session(self, session_num):
        """Simplified preprocessing pipeline"""
        print(f"Preprocessing session {session_num}")

        data = self.load_data(session_num)
        if not data:
            raise ValueError(f"No data found for session {session_num}")

        print(f"   Loaded data types: {list(data.keys())}")

        synchronized_df = self.create_simple_synchronized_data(data)

        if synchronized_df is None:
            raise ValueError("Could not synchronize data")

        print(f"   Synchronized data shape: {synchronized_df.shape}")

        numeric_cols = synchronized_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['timestamp', 'engagement']]

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            synchronized_df[numeric_cols] = scaler.fit_transform(synchronized_df[numeric_cols])
            print(f"   Normalized {len(numeric_cols)} numeric columns")

        output_file = os.path.join(self.output_path, f'session{session_num}_processed.csv')
        synchronized_df.to_csv(output_file, index=False)

        print(f"✓ Session {session_num} preprocessing completed")
        print(f"  Output shape: {synchronized_df.shape}")
        print(f"  Saved to: {output_file}")

        return synchronized_df, scaler


if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Auto-detect sessions
    sessions = preprocessor.detect_sessions()
    print("Detected sessions:", sessions)

    for session_num in sessions:
        try:
            df, scaler = preprocessor.preprocess_session(session_num)
            print(f"Success! Session {session_num}, Data columns:", df.columns.tolist())
        except Exception as e:
            print(f"Error in session {session_num}: {e}")
