import re


def split_with_punctuation(s: str, split_pattern):
    # 匹配以 .,!? 中任意一个符号结尾的片段
    parts = re.findall(r'.*?[{}](?=\s|$)'.format(split_pattern), s)

    remainder = s[sum(len(p) for p in parts) :]
    if remainder:
        parts.append(remainder)
    return parts
