import zipfile
import shutil
from maybe_download import *


if __name__ == '__main__':
    tinyimagenet_base_url = "http://cs231n.stanford.edu/"
    tinyimagenet_filename = "tiny-imagenet-200.zip"
    unzipped_filename = "tiny-imagenet-200"

    prefix = os.path.join("download", "dwr")
    data_prefix = os.path.join("data")

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)
    
    glove_zip = maybe_download(tinyimagenet_base_url, tinyimagenet_filename, prefix, None)

    print("Extracting")
    glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, tinyimagenet_filename), 'r')

    glove_zip_ref.extractall(prefix)
    glove_zip_ref.close()
    
    print("Moving datasets to {}".format(data_prefix))
    shutil.move(os.path.join(prefix, unzipped_filename), os.path.join(data_prefix, unzipped_filename))
