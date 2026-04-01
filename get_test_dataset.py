import os
import pandas as pd


SOURCE_FILE = 'battery_swapping_routing_data.csv'
OUTPUT_FILE = 'battery_swapping_routing_test_dataset.csv'
TEST_ROWS = 20000


def build_test_dataset(source_file, output_file, test_rows=20000):
	"""
	基于原始数据构建测试集：按 datetime 升序后取最后 test_rows 行。
	这样可模拟“用过去预测未来”的评估场景。
	"""
	if not os.path.exists(source_file):
		raise FileNotFoundError(f'未找到源数据文件: {source_file}')

	print(f'读取源数据: {source_file}')
	df = pd.read_csv(source_file)
	print(f'源数据形状: {df.shape}')

	if 'datetime' in df.columns:
		dt_series = pd.to_datetime(df['datetime'], errors='coerce')
		invalid_dt = dt_series.isna().sum()
		if invalid_dt > 0:
			print(f'警告: datetime 有 {invalid_dt} 行无法解析，已按原值参与排序。')
		df = df.assign(_dt_sort=dt_series).sort_values(by=['_dt_sort'], kind='stable').drop(columns=['_dt_sort'])
	else:
		print('警告: 未发现 datetime 字段，将按原始顺序截取末尾样本。')

	if test_rows <= 0:
		raise ValueError('test_rows 必须是正整数。')

	test_rows = min(test_rows, len(df))
	test_df = df.tail(test_rows).copy()
	print(f'测试集形状: {test_df.shape}')

	test_df.to_csv(output_file, index=False)
	print(f'测试集已保存: {output_file}')


if __name__ == '__main__':
	build_test_dataset(SOURCE_FILE, OUTPUT_FILE, TEST_ROWS)
