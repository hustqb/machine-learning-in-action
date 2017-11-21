from random import randrange
import numpy as np


def is_float(s):
	try:
		float(s)
		return True
	except ValueError:
		return False


def load_dataset():
	file_path = '../ensemble/sonar-all-data.txt'
	dataset = []
	with open(file_path, 'r') as f:
		for line in f:
			if line:
				sample = []
				for item in line.split(','):
					item = item.strip()
					if is_float(item):
						sample.append(float(item))
					else:  # 字符=>类标签
						sample.append(item)
				dataset.append(sample)
	return dataset


def split_dataset(dataset, feature, value):
	left, right = [], []
	for sample in dataset:
		if sample[feature] < value:
			left.append(sample)
		else:
			right.append(sample)
	return left, right


def calc_gini(classes, *datasets):
	gini = 0.0
	for d in datasets:
		n_samples = len(d)
		if n_samples == 0:
			continue
		for c in classes:
			proportion = [sample[-1] for sample in d].count(c) / float(n_samples)
			gini += proportion * (1 - proportion)
	return gini


def choose_and_split(dataset, features, classes):
	min_gini = np.inf
	best_f, best_v = 0, 0.0
	best_left, best_right = None, None
	for f in features:
		values = set(sample[f] for sample in dataset)
		for v in values:
			left, right = split_dataset(dataset, f, v)
			gini = calc_gini(classes, left, right)
			if gini < min_gini:
				min_gini = gini
				best_f, best_v = f, v
				best_left, best_right = left, right
	return {'feature': best_f, 'value': best_v, 'left': best_left, 'right': best_right}


def sub_features(dataset, n_features):
	features = set()
	while len(features) < n_features:
		f = randrange(len(dataset[0]) - 1)
		features.add(f)
	return features


def create_tree(dataset, max_depth, min_size, n_features):
	features = sub_features(dataset, n_features)
	classes = set(sample[-1] for sample in dataset)
	root = choose_and_split(dataset, features, classes)
	features.remove(root['feature'])
	create_node(root, features, max_depth, min_size, 1)
	return root


def majority_class(dataset):
	labels = [sample[-1] for sample in dataset]
	lb = max(set(labels), key=labels.count)
	return lb


def create_node(node, features, max_depth, min_size, depth):
	left, right = node['left'], node['right']
	if not left or not right:
		node['left'] = node['right'] = majority_class(left + right)
		return
	if left:
		classes = set(sample[-1] for sample in left)
		# Only one class or no feauture remained
		if (len(classes) == 1 or len(features) == 0 or
			    not features or len(left) < min_size or depth >= max_depth):
			# return classification result
			node['left'] = majority_class(left)
		else:
			node['left'] = choose_and_split(left, features, classes)
			features.remove(node['left']['feature'])
			create_node(node['left'], features,
			            max_depth, min_size, depth + 1)
	if right:
		classes = set(sample[-1] for sample in right)
		if (len(classes) == 1 or len(features) == 0 or
			    not features or len(left) < min_size or depth >= max_depth):
			node['right'] = majority_class(right)
		else:
			node['right'] = choose_and_split(right, features, classes)
			features.remove(node['right']['feature'])
			create_node(node['right'], features,
			            max_depth, min_size, depth + 1)
	return


def sub_samples(dataset, ratio):
	samples = []
	n_sample = round(len(dataset) * ratio)
	while len(samples) < n_sample:
		index = randrange(len(dataset))
		samples.append(dataset[index])
	return list(samples)


def predict(node, sample):
	if sample[node['feature']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], sample)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], sample)
		else:
			return node['right']


def bagging_predict(trees, sample):
	predictions = [predict(tree, sample) for tree in trees]
	return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth=20, min_size=1, sample_size=0.75, n_trees=10, n_features=30):
	"""
	Use random forest and return a prediction.
	train: 训练数据集
	test: 测试数据集
	max_depth: 决策树深度限制，太深容易过拟合
	min_size: 叶子结点大小限制
	sample_size: 单个树的训练集随机采样的比例
	n_trees: 决策树个数
	n_features: 随机选择的特征的个数
	"""
	trees = []
	for i in range(n_trees):
		sub_train = sub_samples(train, sample_size)
		tree = create_tree(sub_train, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, sample) for sample in test]
	return predictions


def accuracy_metric(actual, predictions):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predictions[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def main():
	dataset = load_dataset()
	train_set = dataset[:-10]
	test_set = []
	for sample in dataset[-10:]:
		sample_copy = list(sample)
		sample_copy[-1] = None
		test_set.append(sample_copy)
	predictions = random_forest(train_set, test_set)
	actual_labels = [sample[-1] for sample in dataset[-10:]]
	print(predictions)
	print(actual_labels)
	accuracy = accuracy_metric(actual_labels, predictions)
	print('使用随机森林检测声呐信号的准确率是=>', accuracy)


if __name__ == '__main__':
	main()
