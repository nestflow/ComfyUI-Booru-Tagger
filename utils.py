# Original code: https://github.com/pythongosssss/ComfyUI-WD14-Tagger/blob/main/pysssss.py
import os
import json
import aiohttp
from tqdm import tqdm

config = None


def is_logging_enabled():
    return get_extension_config().get("logging", False)


def log(message, type=None, always=False):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    name = get_extension_config()["name"]

    print(f"(Booru Tagger:{name}) {message}")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(os.path.abspath(__file__))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_extension_config(reload=False):
    global config
    if not reload and config is not None:
        return config

    config_path = get_ext_dir("models.json")

    if not os.path.exists(config_path):
        log("Missing models.json, this extension may not work correctly. Please reinstall the extension.",
            type="ERROR", always=True)
        print(f"Extension path: {get_ext_dir()}")
        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config


def init(check_imports):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.", type="ERROR", always=True)
                return False

    return True


async def download_to_file(url, destination, session=None, progress_cb=None):
    """Stream `url` to `destination`, written atomically via a ``.part`` file.

    Args:
        session: optional aiohttp.ClientSession to reuse, e.g. one that already
            carries the HuggingFace Authorization header. When omitted, a session
            is created and closed inside this function.
        progress_cb: optional ``callable(total, downloaded)`` invoked once after
            the response headers arrive (``downloaded == 0``) and then for every
            downloaded chunk. ``total`` is the server's Content-Length, or 0 when
            the server did not provide one.
    """
    close_session = session is None
    if close_session:
        session = aiohttp.ClientSession()

    proxy = (os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
             or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"))
    proxy_auth = None
    if proxy:
        proxy_auth = aiohttp.BasicAuth(os.getenv("PROXY_USER", ""), os.getenv("PROXY_PASS", ""))

    tmp_destination = destination + ".part"
    try:
        async with session.get(url, proxy=proxy, proxy_auth=proxy_auth) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or 0
            if progress_cb is not None:
                progress_cb(total, 0)

            with tqdm(unit="B", unit_scale=True, miniters=1,
                      desc=url.split("/")[-1], total=total or None) as progressbar:
                with open(tmp_destination, mode="wb") as f:
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progressbar.update(len(chunk))
                        if progress_cb is not None:
                            progress_cb(total, downloaded)

        # Only publish the file once the download fully succeeded.
        os.replace(tmp_destination, destination)
    except BaseException:
        # Never leave a partial file that would later be mistaken for a complete one.
        try:
            os.remove(tmp_destination)
        except OSError:
            pass
        raise
    finally:
        if close_session:
            await session.close()
