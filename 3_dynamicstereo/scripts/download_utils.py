# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import requests
import functools
import json
import warnings

from argparse import ArgumentParser
from typing import List, Optional
from multiprocessing import Pool
from tqdm import tqdm


from dynamic_stereo.scripts.checksum_check import check_dr_sha256


def download_dataset(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_splits: List[str] = ['real', 'valid', 'test', 'train'],
    checksum_check: bool = False,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
    sha256s_file: Optional[str] = None,
):
    """
    Downloads and unpacks the dataset in CO3D format.
    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        the download is finished.
    Args:
        link_list_file: A text file with the list of zip file download links.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        download_splits: A list of data splits to download.
            Must be in ['real', 'valid', 'test', 'train'].
        checksum_check: Enable validation of the downloaded file's checksum before
            extraction.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
    """

    if checksum_check and not sha256s_file:
        raise ValueError(
            "checksum_check is requested but ground-truth SHA256 file not provided!"
        )

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a json"
            " with zip file download links."
            # " The file is stored in the DynamicStereo github:"
            # " https://github.com/facebookresearch/dynamic_stereo/blob/main/dynamic_stereo/links.json"
        )

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    # read the link file
    with open(link_list_file, "r") as f:
        links = json.load(f)

    for split in download_splits:
        if split not in ['real', 'valid', 'test', 'train']:
            raise ValueError(
                        f"Download split {str(split)} is not valid"
                    )

    data_links = []
    for split_name, urls in links.items():
        if split_name in download_splits:
            for url in urls:
                link_name = os.path.split(url)[-1]
                data_links.append((split_name, link_name, url))


    with Pool(processes=n_download_workers) as download_pool:
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_split_file,
                    download_folder,
                    checksum_check,
                    sha256s_file,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_split_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass
    print("Done")



def build_arg_parser(
    dataset_name: str,
    default_link_list_file: str,
    default_sha256_file: str,
) -> ArgumentParser:
    parser = ArgumentParser(description=f"Download the {dataset_name} dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--download_splits",
        default=['real', 'valid', 'test', 'train'],
        nargs='+',
        help=f"A comma-separated list of {dataset_name} splits to download.",
    )
    parser.add_argument(
        "--link_list_file",
        type=str,
        default=default_link_list_file,
        help=(
            f"The file with html links to the {dataset_name} dataset files."
            + " In most cases the default local file `links.json` should be used."
        ),
    )
    parser.add_argument(
        "--sha256_file",
        type=str,
        default=default_sha256_file,
        help=(
            f"The file with SHA256 hashes of {dataset_name} dataset files."
            + " In most cases the default local file `dr_sha256.json` should be used."
        ),
    )
    parser.add_argument(
        "--checksum_check",
        action="store_true",
        default=True,
        help="Check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.add_argument(
        "--no_checksum_check",
        action="store_false",
        dest="checksum_check",
        default=False,
        help="Does not check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.set_defaults(checksum_check=True)
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )

    return parser

def _unpack_split_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    split, link_name, url = link
    local_fl = os.path.join(download_folder, link_name)
    print(f"Unpacking dataset file {local_fl} ({link_name}) to {download_folder}.")
    
    download_folder_split = os.path.join(download_folder, split)
    # os.makedirs(download_folder_split, exist_ok=True)
    shutil.unpack_archive(local_fl, download_folder_split)
    if clear_archive:
        os.remove(local_fl)

def _download_split_file(
    download_folder: str,
    checksum_check: bool,
    sha256s_file: Optional[str],
    skip_downloaded_files: bool,
    link: str,
):
    __, link_name, url = link
    local_fl_final = os.path.join(download_folder, link_name)

    if skip_downloaded_files and os.path.isfile(local_fl_final):
        print(f"Skipping {local_fl_final}, already downloaded!")
        return link_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    local_fl = os.path.join(in_progress_folder, link_name)

    print(f"Downloading dataset file {link_name} ({url}) to {local_fl}.")
    _download_with_progress_bar(url, local_fl, link_name)
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        try:
            check_dr_sha256(
                local_fl,
                sha256s_file=sha256s_file,
            )
        except AssertionError:
            warnings.warn(
                f"Checksums for {local_fl} did not match!"
                + " This is likely due to a network failure,"
                + " please restart the download script." 
            )
            return link_name, False
        
    os.rename(local_fl, local_fl_final)
    return link_name, True


def _download_with_progress_bar(url: str, fname: str, filename: str):

    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max((max(total // 1024, 1) // 20), 1) == 0:
                print(f"{filename}: Downloaded {100.0*(float(bar.n)/max(total, 1)):3.1f}%.")
                print(bar)