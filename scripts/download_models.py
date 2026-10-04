"""Resume official BGE downloads by range; verify SHA256 before caching weights."""

import concurrent.futures
import hashlib
import ipaddress
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import requests

# Optional route diagnostic; HTTPS still verifies the original CDN hostname.
cdn_ipv4 = os.getenv("BGE_CDN_IPV4")
if cdn_ipv4:
    ipaddress.IPv4Address(cdn_ipv4)
    original_getaddrinfo = socket.getaddrinfo

    def cdn_getaddrinfo(host, port, *args, **kwargs):
        if host == "us.aws.cdn.hf.co":
            host = cdn_ipv4
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = cdn_getaddrinfo

CACHE = Path.home() / ".cache/huggingface/hub"


def model(repo, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    response = requests.head(url, allow_redirects=False, timeout=30)
    response.raise_for_status()
    size = int(response.headers["x-linked-size"])
    digest = response.headers["x-linked-etag"].strip('"')
    destination = CACHE / ("models--" + repo.replace("/", "--")) / "blobs" / digest
    if destination.exists():
        print(repo, "already cached", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    refresh_lock = threading.Lock()
    thread_state = threading.local()
    parts_dir = (
        Path(__file__).resolve().parents[1] / ".cache/downloads"
    ) / repo.replace("/", "--")
    parts_dir.mkdir(parents=True, exist_ok=True)
    block = 1024 * 1024
    count = (size + block - 1) // block

    def download(i):
        nonlocal response
        start, end = i * block, min((i + 1) * block, size) - 1
        part = parts_dir / str(i)
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        if not hasattr(thread_state, "session"):
            thread_state.session = requests.Session()
        partial = parts_dir / f"{i}.partial"
        for attempt in range(20):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == end - start + 1:
                    os.replace(partial, part)
                    return
                if offset > end - start + 1:
                    partial.unlink()
                    offset = 0
                range_start = start + offset
                download_url = response.headers["location"]
                r = thread_state.session.get(
                    download_url,
                    headers={"Range": f"bytes={range_start}-{end}"},
                    timeout=(30, 90),
                    stream=True,
                )
                if r.status_code == 403:
                    r.close()
                    with refresh_lock:
                        if response.headers["location"] != download_url:
                            continue
                        response = requests.head(
                            url + f"?refresh={int(time.time())}",
                            allow_redirects=False,
                            timeout=30,
                        )
                        response.raise_for_status()
                        if response.headers["x-linked-etag"].strip('"') != digest:
                            raise ValueError(
                                "Upstream model revision changed during download"
                            )
                    continue
                r.raise_for_status()
                if (
                    r.status_code != 206
                    or r.headers.get("Content-Range")
                    != f"bytes {range_start}-{end}/{size}"
                ):
                    raise ValueError("Unexpected range response")
                try:
                    with partial.open("ab") as target:
                        for data in r.iter_content(chunk_size=64 * 1024):
                            target.write(data)
                finally:
                    r.close()
                if partial.stat().st_size != end - start + 1:
                    raise ValueError("Unexpected response length")
                os.replace(partial, part)
                return
            except (requests.RequestException, ValueError, OSError) as exc:
                if attempt in (0, 9, 19):
                    print(
                        f"Retry {repo} part={i} attempt={attempt + 1} error={type(exc).__name__}",
                        flush=True,
                    )
                if attempt == 19:
                    raise
                time.sleep(min(attempt + 1, 10))
        raise RuntimeError(f"Range download failed: {repo} part {i}")

    print(repo, size, "bytes;", count, "parts", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for completed, _ in enumerate(pool.map(download, range(count)), 1):
            if completed % 10 == 0:
                print(repo, completed, "/", count, flush=True)
    temporary = destination.with_suffix(".parallel")
    hasher = hashlib.sha256()
    with temporary.open("wb") as out:
        for i in range(count):
            data = (parts_dir / str(i)).read_bytes()
            hasher.update(data)
            out.write(data)
    if hasher.hexdigest() != digest:
        temporary.unlink()
        raise ValueError("Model SHA256 mismatch")
    os.replace(temporary, destination)
    shutil.rmtree(parts_dir)
    print(repo, "SHA256 verified, cached", flush=True)


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda item: model(*item),
                [
                    ("BAAI/bge-m3", "pytorch_model.bin"),
                    ("BAAI/bge-reranker-v2-m3", "model.safetensors"),
                ],
            )
        )


if __name__ == "__main__":
    main()
