"""
Utility functions for calculating goodput metrics from job execution data.

Goodput measures the efficiency of job execution by comparing actual training time
to total elapsed time (including queueing and overhead).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def _convert_to_timezone_naive(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Convert a datetime column to timezone-naive datetime.
    
    Handles both timezone-aware and timezone-naive datetime strings by first
    converting to UTC, then removing timezone information.
    
    Args:
        df: DataFrame containing the column
        column: Name of the datetime column to convert
        
    Returns:
        Series with timezone-naive datetime values
    """
    return pd.to_datetime(df[column], format='ISO8601', utc=True).dt.tz_localize(None)


def calculate_queueing_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate queueing time for each job in minutes.
    
    Queueing time = metadata.started - metadata.submission_time
    
    Args:
        df: DataFrame with columns 'metadata.submission_time' and 'metadata.started'
        
    Returns:
        DataFrame with additional column 'metadata.queueing_time' (in minutes)
    """
    df = df.copy()
    
    # Convert timestamps to timezone-naive datetime
    df['metadata.submission_time'] = _convert_to_timezone_naive(df, 'metadata.submission_time')
    df['metadata.started'] = _convert_to_timezone_naive(df, 'metadata.started')
    
    # Calculate queueing time in minutes
    df['metadata.queueing_time'] = (
        df['metadata.started'] - df['metadata.submission_time']
    ).dt.total_seconds() / 60
    
    return df


def calculate_per_job_goodput(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-job goodput as the ratio of training time to total elapsed time.
    
    Per-job goodput = metadata.train_runtime / (metadata.completed - metadata.submission_time)
    
    This metric represents the fraction of total time spent on actual training,
    with values between 0 and 1 (or 0-100%).
    
    Args:
        df: DataFrame with columns 'metadata.train_runtime', 'metadata.completed',
            and 'metadata.submission_time'
            
    Returns:
        DataFrame with additional column 'metadata.per_job_goodput'
    """
    df = df.copy()
    
    # Convert timestamps to timezone-naive datetime
    df['metadata.submission_time'] = _convert_to_timezone_naive(df, 'metadata.submission_time')
    df['metadata.completed'] = _convert_to_timezone_naive(df, 'metadata.completed')
    df['metadata.started'] = _convert_to_timezone_naive(df, 'metadata.started')
    
    # Calculate total elapsed time in seconds
    total_elapsed_time = (
        df['metadata.completed'] - df['metadata.submission_time']
    ).dt.total_seconds()
    
    # Calculate training time as difference between completed and started
    train_time = (
        df['metadata.completed'] - df['metadata.started']
    ).dt.total_seconds()
    
    # Calculate per-job goodput
    # If metadata.completed or metadata.started is None, or train_time <= 0, set goodput to 0
    # Otherwise: train_time / total_elapsed_time
    df['metadata.per_job_goodput'] = np.where(
        df['metadata.completed'].isna() | df['metadata.started'].isna() | (train_time <= 0),
        0.0,
        np.where(
            total_elapsed_time > 0,
            train_time / total_elapsed_time,
            0.0
        )
    )
    
    return df


def calculate_overall_goodput_with_failed_jobs(df: pd.DataFrame) -> float:
    """
    Calculate overall scheduling goodput across all jobs (including failed/incomplete).
    
    Overall goodput = sum(metadata.train_runtime) / sum(end_time - metadata.submission_time)
    
    Where end_time is metadata.completed if available, otherwise metadata.last_updated.
    
    This represents the cluster-wide efficiency, showing what fraction of total
    elapsed time across all jobs was spent on actual training.
    
    Args:
        df: DataFrame with goodput calculations already performed
        
    Returns:
        Float representing overall goodput (0 to 1)
    """
    # Use all jobs, not just completed ones
    # Filter only for valid numeric values and valid timestamps
    df_copy = df.copy()
    
    # Convert timestamps if not already done
    if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.submission_time']):
        df_copy['metadata.submission_time'] = _convert_to_timezone_naive(df_copy, 'metadata.submission_time')
    if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.started']):
        df_copy['metadata.started'] = _convert_to_timezone_naive(df_copy, 'metadata.started')
    
    # Use metadata.completed if available, otherwise fall back to metadata.last_updated
    if 'metadata.last_updated' in df_copy.columns:
        # Convert both timestamp columns
        if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.completed']):
            df_copy['metadata.completed'] = _convert_to_timezone_naive(df_copy, 'metadata.completed')
        if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.last_updated']):
            df_copy['metadata.last_updated'] = _convert_to_timezone_naive(df_copy, 'metadata.last_updated')
        
        # Create end_time column: use completed if available, otherwise last_updated
        df_copy['end_time'] = df_copy['metadata.completed'].fillna(df_copy['metadata.last_updated'])
    else:
        # If last_updated doesn't exist, just use completed
        if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.completed']):
            df_copy['metadata.completed'] = _convert_to_timezone_naive(df_copy, 'metadata.completed')
        df_copy['end_time'] = df_copy['metadata.completed']
    
    # Calculate elapsed time using end_time
    elapsed_time = (df_copy['end_time'] - df_copy['metadata.submission_time']).dt.total_seconds()
    
    
    # Filter for valid jobs
    valid_jobs = df_copy[
        df_copy['metadata.train_runtime'].notna() &
        (df_copy['metadata.train_runtime'] >= 0) &
        (elapsed_time >= 0)
    ].copy()

    print(f"Valid jobs: {len(valid_jobs)}")
    print(f"Total jobs: {len(df_copy)}")
    print(f"Invalid jobs: {len(df_copy) - len(valid_jobs)}")

    
    if len(valid_jobs) == 0:
        return 0.0
    
    # Calculate total training time and total elapsed time
    total_train_runtime = (
        valid_jobs['metadata.completed'] - valid_jobs['metadata.started']
    ).dt.total_seconds().sum()
    total_elapsed_time = (
        df_copy['end_time'] - df_copy['metadata.submission_time']
    ).dt.total_seconds().sum()
    
    if total_elapsed_time <= 0:
        return 0.0
    
    return total_train_runtime / total_elapsed_time

def calculate_overall_goodput(df: pd.DataFrame) -> float:
    """
    Calculate overall scheduling goodput across all jobs (including failed/incomplete).
    
    Overall goodput = sum(metadata.train_runtime) / sum(metadata.completed - metadata.submission_time)
    
    This represents the cluster-wide efficiency, showing what fraction of total
    elapsed time across all jobs was spent on actual training.
    
    Args:
        df: DataFrame with goodput calculations already performed
        
    Returns:
        Float representing overall goodput (0 to 1)
    """
    # Use all jobs, not just completed ones
    # Filter only for valid numeric values and valid timestamps
    df_copy = df.copy()
    
    # Convert timestamps if not already done
    if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.submission_time']):
        df_copy['metadata.submission_time'] = _convert_to_timezone_naive(df_copy, 'metadata.submission_time')
    if not pd.api.types.is_datetime64_any_dtype(df_copy['metadata.completed']):
        df_copy['metadata.completed'] = _convert_to_timezone_naive(df_copy, 'metadata.completed')
    
    # Calculate elapsed time
    elapsed_time = (df_copy['metadata.completed'] - df_copy['metadata.submission_time']).dt.total_seconds()
    
    # Filter for valid jobs
    valid_jobs = df_copy[
        df_copy['metadata.train_runtime'].notna() &
        (df_copy['metadata.train_runtime'] >= 0) &
        (elapsed_time > 0)
    ].copy()
    
    if len(valid_jobs) == 0:
        return 0.0
    
    # Calculate total training time and total elapsed time
    total_train_runtime = valid_jobs['metadata.train_runtime'].sum()
    total_elapsed_time = (
        valid_jobs['metadata.completed'] - valid_jobs['metadata.submission_time']
    ).dt.total_seconds().sum()
    
    if total_elapsed_time <= 0:
        return 0.0
    
    return total_train_runtime / total_elapsed_time

def identify_invalid_elapsed_time_jobs(df: pd.DataFrame, output_csv: str = 'invalid_elapsed_time_jobs.csv') -> pd.DataFrame:
    """
    Identify and save jobs where (completed - submission_time) <= 0.
    
    These are jobs with invalid elapsed time, which could indicate:
    - Data quality issues
    - Clock synchronization problems
    - Jobs that completed before they were submitted (impossible)
    
    Args:
        df: DataFrame with timestamp columns
        output_csv: Path to save the invalid jobs CSV
        
    Returns:
        DataFrame containing only the invalid jobs
    """
    # Ensure timestamps are converted
    df = df.copy()
    if 'metadata.submission_time' in df.columns and 'metadata.completed' in df.columns:
        df['metadata.submission_time'] = _convert_to_timezone_naive(df, 'metadata.submission_time')
        df['metadata.completed'] = _convert_to_timezone_naive(df, 'metadata.completed')
        
        # Calculate elapsed time
        elapsed_time = (df['metadata.completed'] - df['metadata.submission_time']).dt.total_seconds()
        
        # Find invalid jobs
        invalid_jobs = df[elapsed_time <= 0].copy()
        invalid_jobs['elapsed_time_seconds'] = elapsed_time[elapsed_time <= 0]
        
        if len(invalid_jobs) > 0:
            invalid_jobs.to_csv(output_csv, index=False)
            print(f"Found {len(invalid_jobs)} jobs with invalid elapsed time (completed - submission <= 0)")
            print(f"Saved to: {output_csv}")
        else:
            print("No jobs found with invalid elapsed time")
        
        return invalid_jobs
    else:
        print("Required timestamp columns not found")
        return pd.DataFrame()


def get_goodput_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for per-job goodput.
    
    Args:
        df: DataFrame with 'metadata.per_job_goodput' column
        
    Returns:
        Dictionary containing:
        - count: number of valid jobs
        - mean, median, std: basic statistics
        - min, max: range
        - p25, p50, p75, p95, p99: percentiles
        - filtered_count: number of jobs filtered out
    """
    # Filter to valid goodput values
    valid_goodput = df[
        df['metadata.per_job_goodput'].notna() &
        (df['metadata.per_job_goodput'] > 0) &
        (df['metadata.per_job_goodput'] <= 1)  # Goodput should be <= 1
    ]['metadata.per_job_goodput']

    if len(valid_goodput) == 0:
        return {
            'count': 0,
            'filtered_count': len(df),
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'p25': np.nan,
            'p50': np.nan,
            'p75': np.nan,
            'p95': np.nan,
            'p99': np.nan
        }
    
    return {
        'count': len(valid_goodput),
        'filtered_count': len(df) - len(valid_goodput),
        'mean': valid_goodput.mean(),
        'median': valid_goodput.median(),
        'std': valid_goodput.std(),
        'min': valid_goodput.min(),
        'max': valid_goodput.max(),
        'p25': valid_goodput.quantile(0.25),
        'p50': valid_goodput.quantile(0.50),
        'p75': valid_goodput.quantile(0.75),
        'p95': valid_goodput.quantile(0.95),
        'p99': valid_goodput.quantile(0.99)
    }


def process_goodput_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    """
    Convenience function to load CSV and calculate all goodput metrics.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (processed_dataframe, statistics_dict, overall_goodput)
    """
    df = pd.read_csv(csv_path)
    df = calculate_queueing_time(df)
    df = calculate_per_job_goodput(df)
    stats = get_goodput_statistics(df)
    overall = calculate_overall_goodput(df)
    
    return df, stats, overall

# Made with Bob
