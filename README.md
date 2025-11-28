Adapted from Sajjad's notebook

You'll want to install the artemis package and run:

python artemis/scripts/preprocess_artemis_data.py \
     -save-out-dir ./processed_artemis \
     -raw-artemis-data-csv ./official_data/artemis_dataset_release_v0.csv \
     --preprocess-for-deep-nets True

to create the artemis_preprocessed.csv table.

