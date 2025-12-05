import time
import io
import requests
from PIL import Image

GOPRO_IP = "10.5.5.9"
CONTROL_BASE = f"http://{GOPRO_IP}"
MEDIA_BASE = f"http://{GOPRO_IP}:8080"

def send_command(path, desc=None):
    url = f"{CONTROL_BASE}{path}"
    if desc:
        print(desc, "->", url)
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r

def get_media_list():
    url = f"{MEDIA_BASE}/gp/gpMediaList"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()

def newest_photo_from_media_list(media_json):
    """
    Returns (folder_name, file_name) for the newest JPG it can find.
    GoPro returns folders under media["media"], each with "fs" file entries.
    """
    media = media_json.get("media", [])
    best = None  # (sort_key, folder, filename)

    for folder in media:
        folder_name = folder.get("d")
        for f in folder.get("fs", []):
            name = f.get("n", "")
            if not name.lower().endswith((".jpg", ".jpeg")):
                continue

            # Prefer timestamp if present, else fallback to filename
            # Some firmwares use "mod" (modified time) or "cre" (created)
            sort_key = f.get("mod") or f.get("cre") or name
            cand = (sort_key, folder_name, name)
            if best is None or cand[0] > best[0]:
                best = cand

    if not best:
        return None
    return best[1], best[2]

def download_photo(folder, filename):
    # Photos are served under /videos/DCIM/...
    url = f"{MEDIA_BASE}/videos/DCIM/{folder}/{filename}"
    print("Downloading:", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def main():
    # 1) Basic reachability
    try:
        print("Checking connection to GoPro...")
        requests.get(f"{CONTROL_BASE}/gp/gpControl", timeout=5).raise_for_status()
        print("GoPro reachable.")
    except Exception as e:
        print("Could not reach GoPro API:", e)
        return

    # 2) Switch to photo mode + single photo
    try:
        send_command("/gp/gpControl/command/mode?p=1", "Setting mode to PHOTO")
        time.sleep(0.5)
        send_command("/gp/gpControl/command/sub_mode?mode=1&sub_mode=0", "Setting sub-mode to SINGLE PHOTO")
        time.sleep(0.5)
    except Exception as e:
        print("Failed to set mode/sub-mode:", e)
        return

    # 3) Snapshot media state BEFORE
    try:
        before = get_media_list()
        before_latest = newest_photo_from_media_list(before)
        print("Latest photo before:", before_latest)
    except Exception as e:
        print("Failed to read media list:", e)
        return

    # 4) Trigger shutter
    try:
        send_command("/gp/gpControl/command/shutter?p=1", "Triggering shutter")
        time.sleep(0.2)
        send_command("/gp/gpControl/command/shutter?p=0", "Stopping shutter")
    except Exception as e:
        print("Failed to trigger shutter:", e)
        return

    # 5) Poll until a NEW jpg appears
    newest = None
    for _ in range(15):  # ~15 * 1s = 15s max
        try:
            media = get_media_list()
            newest = newest_photo_from_media_list(media)
            if newest and newest != before_latest:
                break
        except Exception:
            pass
        time.sleep(1)

    if not newest or newest == before_latest:
        print("Timed out waiting for new photo to appear in media list.")
        print("Newest seen:", newest)
        return

    folder, filename = newest
    print("New photo:", newest)

    # 6) Download + display
    try:
        data = download_photo(folder, filename)
        img = Image.open(io.BytesIO(data))
        img.show()  # opens default image viewer
        # optionally save locally:
        out_name = filename
        with open(out_name, "wb") as f:
            f.write(data)
        print("Saved to:", out_name)
    except Exception as e:
        print("Failed to download/display:", e)

if __name__ == "__main__":
    main()
