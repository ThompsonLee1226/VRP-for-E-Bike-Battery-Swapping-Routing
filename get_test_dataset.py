import os
import pandas as pd


SOURCE_FILE = 'battery_swapping_routing_data.csv'
OUTPUT_FILE = 'battery_swapping_routing_test_dataset.csv'
TEST_ROWS = 20000


def build_test_dataset(source_file, output_file, test_rows=20000):
	"""
	Build a test set from raw data: sort by datetime ascending,
	then take the last test_rows rows to simulate future prediction.
	"""
	if not os.path.exists(source_file):
		raise FileNotFoundError(f'Source data file not found: {source_file}')

	print(f'Reading source data: {source_file}')
	df = pd.read_csv(source_file)
	print(f'Source data shape: {df.shape}')

	if 'datetime' in df.columns:
		dt_series = pd.to_datetime(df['datetime'], errors='coerce')
		invalid_dt = dt_series.isna().sum()
		if invalid_dt > 0:
			print(f'Warning: {invalid_dt} datetime values could not be parsed; original ordering is kept for those rows.')
		df = df.assign(_dt_sort=dt_series).sort_values(by=['_dt_sort'], kind='stable').drop(columns=['_dt_sort'])
	else:
		print('Warning: datetime column not found, tail rows will be selected in original order.')

	if test_rows <= 0:
		raise ValueError('test_rows must be a positive integer.')

	test_rows = min(test_rows, len(df))
	test_df = df.tail(test_rows).copy()
	print(f'Test set shape: {test_df.shape}')

	test_df.to_csv(output_file, index=False)
	print(f'Test set saved to: {output_file}')


if __name__ == '__main__':
	build_test_dataset(SOURCE_FILE, OUTPUT_FILE, TEST_ROWS)
