echo "Starting the training and inferencing script"
python main.py
echo "Running tests on the inferencing script"
python -m unittest tests/test_batch_predict.py