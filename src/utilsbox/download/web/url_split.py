def url_spilt(url: str):
    parts = url.split("/")
    parts[0] = parts[0] + "//" + parts[2]
    del parts[1:3]
    return parts