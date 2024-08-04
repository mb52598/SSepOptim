from typing import Callable

import requests


def gdrive_download(
    file_id: str, path: str, report_hook: Callable[[int, int, int], None]
):
    gdrive_url = (
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    )
    gdrive_url_confirm = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    session = requests.Session()

    response = session.get(gdrive_url, stream=True)

    if not response.ok:
        raise RuntimeError(f"Unable to fetch url: {gdrive_url}")

    # Virus warning, we need to confirm
    if "text/html" in response.headers["Content-Type"]:
        response = session.get(gdrive_url_confirm, stream=True)

        if not response.ok:
            raise RuntimeError(f"Unable to fetch url: {gdrive_url_confirm}")

    block_size = 1024 * 32
    block_number = 0
    content_length_header_key = "Content-Length"
    content_length = (
        int(response.headers[content_length_header_key])
        if content_length_header_key in response.headers
        else -1
    )
    with open(path, "xb") as file:
        for chunk in response.iter_content(block_size):
            if chunk:
                file.write(chunk)
                block_number += 1
                report_hook(block_number, block_size, content_length)
