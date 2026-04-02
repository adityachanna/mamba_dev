from urllib.parse import unquote, urlparse

from s3_upload import get_s3_client

# URL from Cloudflare R2. This is usually private, so direct HTTP may return 400/Authorization errors.
url = "https://b93fe5d8a511add2140e2fe05b83e831.r2.cloudflarestorage.com/bucketprod/JDI/problems/req_8f3a21/001_Screenshot 2026-03-27 110048.png"


def parse_r2_url(r2_url: str) -> tuple[str, str]:
    parsed = urlparse(r2_url)
    if not parsed.path or parsed.path == "/":
        raise ValueError("R2 URL does not contain bucket and key")

    path = parsed.path.lstrip("/")
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError("R2 URL path must include bucket and object key")

    bucket_name, object_key = parts[0], unquote(parts[1])
    return bucket_name, object_key


def download_image_from_r2(r2_url: str, target_file: str = "screenshot.png") -> None:
    bucket_name, object_key = parse_r2_url(r2_url)

    client = get_s3_client()
    obj = client.get_object(Bucket=bucket_name, Key=object_key)
    content = obj["Body"].read()

    with open(target_file, "wb") as f:
        f.write(content)

    print(f"Downloaded image to {target_file} from bucket={bucket_name}, key={object_key}")


if __name__ == "__main__":
    try:
        download_image_from_r2(url)
    except Exception as exc:
        print("Failed to retrieve image:", exc)

