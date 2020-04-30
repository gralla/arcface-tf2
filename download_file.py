from google_drive_downloader import GoogleDriveDownloader as gdd

#This will download a mnist.zip file into a data folder and unzip it
#Set showsize=True to see the download progress
#Set overwrite=True you really want to overwrite an already existent file.

gdd.download_file_from_google_drive(file_id='1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp',
                                    dest_path='/data/datasets/test_datasets/agedb_align_112.zip',
                                    unzip=True,
                                    showsize=True,
                                    overwrite=True)