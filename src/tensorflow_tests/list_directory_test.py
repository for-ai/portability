import os.path

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class PathLike(object):
  """Backport of pathlib.Path for Python < 3.6"""

  def __init__(self, name):
    self.name = name

  def __fspath__(self):
    return self.name

  def __str__(self):
    return self.name


run_all_path_types = parameterized.named_parameters(
    ("str", file_io.join),
    ("pathlike", lambda *paths: PathLike(file_io.join(*paths))))


class FileIoTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._base_dir = file_io.join(self.get_temp_dir(), "base_dir")
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)

  @run_all_path_types
  def testListDirectory(self, join):
    dir_path = join(self._base_dir, "test_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = join(str(dir_path), name)
      file_io.FileIO(file_path, mode="w").write("testing")
    subdir_path = join(str(dir_path), "sub_dir")
    file_io.create_dir(subdir_path)
    subdir_file_path = join(str(subdir_path), "file4.txt")
    file_io.FileIO(subdir_file_path, mode="w").write("testing")
    dir_list = file_io.list_directory(dir_path)
    self.assertItemsEqual(files + ["sub_dir"], dir_list)

  def testListDirectoryFailure(self):
    dir_path = file_io.join(self._base_dir, "test_dir")
    with self.assertRaises(errors.NotFoundError):
      file_io.list_directory(dir_path)



if __name__ == "__main__":
      test.main()