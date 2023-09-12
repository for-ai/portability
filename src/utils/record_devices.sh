mkdir ./device_logging
# iterate all files in the /src/tensorflow_tests_reduced directory ending in _test.py
for f in ./src/tensorflow_tests_reduced/*_test.py; do
    # get the filename without the extension
    filename=$(basename "$f")
    filename="${filename%.*}"
    # run the test and record the devices used
    DEVICE=gpu poetry run python src/tensorflow_test.py src/tensorflow_tests_reduced/"$filename".py > ./device_logging/$filename.txt
done