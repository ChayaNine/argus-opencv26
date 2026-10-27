"""Tiny S3 helper used by the remote scripts (avoids shell-quoting gymnastics).

    python deploy/s3tool.py put  <file> <bucket> <key>
    python deploy/s3tool.py get  <bucket> <key> <file>
    python deploy/s3tool.py pull <bucket> <prefix> <dir> [suffix]
"""
import os
import sys

import boto3


def main(argv):
    s3 = boto3.client("s3")
    cmd = argv[0]
    if cmd == "put":
        f, b, k = argv[1:4]
        if os.path.exists(f):
            s3.upload_file(f, b, k)
            print("published", k)
        else:
            print("skip (missing)", f)
    elif cmd == "get":
        b, k, f = argv[1:4]
        s3.download_file(b, k, f)
        print("downloaded", k)
    elif cmd == "pull":
        b, prefix, d = argv[1:4]
        suffix = argv[4] if len(argv) > 4 else ""
        os.makedirs(d, exist_ok=True)
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=prefix):
            for o in page.get("Contents", []):
                k = o["Key"]
                if k.endswith(suffix):
                    s3.download_file(b, k, os.path.join(d, k.split("/")[-1]))
                    print("pulled", k)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
