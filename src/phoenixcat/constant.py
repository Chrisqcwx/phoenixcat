import os
import hashlib

package_name = 'phoenixcat'
package_name_upper = package_name.upper()

USER_NAME = os.getlogin()
USER_NAME_HASH = hashlib.md5(USER_NAME.encode()).hexdigest()

default_home = os.path.join(os.path.expanduser("~"), ".cache", package_name)
HOME = os.path.expanduser(
    os.getenv(
        f"{package_name_upper}_HOME",
        default_home,
    )
)


default_assets_cache_path = os.path.join(HOME, "assets")

ASSETS_CACHE = os.getenv(
    f"{package_name_upper}_ASSETS_CACHE", default_assets_cache_path
)

os.makedirs(ASSETS_CACHE, exist_ok=True)


default_config_path = os.path.join(HOME, "config")

CONFIG_HOME = os.getenv(f"{package_name_upper}_CONFIG_PATH", default_config_path)

os.makedirs(CONFIG_HOME, exist_ok=True)
