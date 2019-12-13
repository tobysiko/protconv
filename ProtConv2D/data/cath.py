from ProtConv2D.utils.data_utils import download_file

from absl import logging

def download_cath(destination):
    assert os.path.exists(destination), f"Path {destination} does not exist"
    cath_base_url = "http://download.cathdb.info/cath/releases/latest-release/"
    cath_file_list = [
        "cath-classification-data/cath-names.txt",
        "cath-classification-data/cath-domain-list.txt",
        "non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz",
        "non-redundant-data-sets/cath-dataset-nonredundant-S40.list",
        "non-redundant-data-sets/cath-dataset-nonredundant-S40.fa"
    ]

    try:
        for f in cath_file_list:
            url = os.path.join(cath_base_url, cath_file_list)
            assert download_file(url, destination), f"Could not download {url}"
    except AssertionError as a:
        logging.error(a)

