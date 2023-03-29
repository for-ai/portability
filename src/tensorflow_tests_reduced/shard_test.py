"""Tests for `tf.data.Dataset.shard()`."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


class ShardTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleCase(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
      dataset = dataset.shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[2, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedData(self):
    dataset_a = dataset_ops.Dataset.range(10)
    dataset_b = dataset_ops.Dataset.range(10, 0, -1)
    with tensorflow_op_timer():
      dataset = dataset_ops.Dataset.zip((dataset_a, dataset_b)).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[(2, 8), (7, 3)])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetZero(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
      dataset = dataset.shard(5, 0)
    self.assertDatasetProduces(dataset, expected_output=[0, 5])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetGreaterNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10)
      with tensorflow_op_timer():
        dataset = dataset.shard(5, 7)
        
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeOffset(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10)
      with tensorflow_op_timer():
        dataset = dataset.shard(5, -3)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10)
      with tensorflow_op_timer():
        dataset = dataset.shard(-3, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testZeroNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(0, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testIteratorEndsBeforeFirstElem(self):
    dataset = dataset_ops.Dataset.range(1)
    with tensorflow_op_timer():
        dataset = dataset.shard(5, 2)
    dataset = dataset_ops.Dataset.range(1).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[])

  @combinations.generate(test_base.default_test_combinations())
  def testLargerWorkerPool(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(7, 5)
    dataset = dataset_ops.Dataset.range(10).shard(7, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(5, 4)
    dataset = dataset_ops.Dataset.range(10).shard(5, 4)
    self.assertDatasetProduces(dataset, expected_output=[4, 9])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards2(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(4, 3)
    dataset = dataset_ops.Dataset.range(10).shard(4, 3)
    self.assertDatasetProduces(dataset, expected_output=[3, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsLargerThanDataset(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(20, 5)
    dataset = dataset_ops.Dataset.range(10).shard(20, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(1, 0, name="shard")
    dataset = dataset_ops.Dataset.from_tensors(42).shard(1, 0, name="shard")
    self.assertDatasetProduces(dataset, [42])


class ShardCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):

  def _build_dataset(self, num_elements, num_shards, index):
    return dataset_ops.Dataset.range(num_elements).shard(num_shards, index)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              elems=[10, 100], num_shards=[2, 5], index=[0, 1])))
  def test(self, verify_fn, elems, num_shards, index):
    verify_fn(
        self,
        lambda: self._build_dataset(elems, num_shards, index),
        num_outputs=elems // num_shards)


class ShardRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3, 4])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.range(4)
    with tensorflow_op_timer():
        dataset = dataset.shard(num_shards=2, index=0)
    dataset = dataset_ops.Dataset.range(4).shard(num_shards=2, index=0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([])
    with tensorflow_op_timer():
        dataset = dataset.shard(
        num_shards=2, index=1)
    dataset = dataset_ops.Dataset.from_tensor_slices([]).shard(
        num_shards=2, index=1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=0))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexLessThanNumElements(self):
    dataset = dataset_ops.Dataset.range(10)
    with tensorflow_op_timer():
        dataset = dataset.shard(5, 0)
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertEqual(0, self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual(5, self.evaluate(random_access.at(dataset, 1)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsGreaterThanNumElementsIndexLess(self):
    dataset = dataset_ops.Dataset.range(7)
    with tensorflow_op_timer():
        dataset = dataset.shard(8, 3)
    dataset = dataset_ops.Dataset.range(7).shard(8, 3)
    self.assertEqual(3, self.evaluate(random_access.at(dataset, 0)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexGreaterThanNumElements(self):
    dataset = dataset_ops.Dataset.range(13)
    with tensorflow_op_timer():
        dataset = dataset.shard(23, 21)
    dataset = dataset_ops.Dataset.range(13).shard(23, 21)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elements=[0, 10, 50],
              num_shards=[5, 7, 10],
              index=[0, 1, 2, 3, 4],
          )))
  def testMultipleCombinations(self, elements, num_shards, index):
    components = range(elements)
    dataset = dataset_ops.Dataset.range(elements)
    with tensorflow_op_timer():
        dataset = dataset.shard(
        num_shards=num_shards, index=index)
    dataset = dataset_ops.Dataset.range(elements).shard(
        num_shards=num_shards, index=index)
    len_dataset = self.evaluate(dataset.cardinality())
    for i in range(self.evaluate(dataset.cardinality())):
      self.assertAllEqual(components[index + (num_shards * i)],
                          self.evaluate(random_access.at(dataset, i)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


if __name__ == "__main__":
  test.main()
