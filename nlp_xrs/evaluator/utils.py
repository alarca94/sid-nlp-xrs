import re


def feature_detect(texts, feature_set):
    p = re.compile(f"\\b({'|'.join(feature_set)})\\b")
    return [set(p.findall(t)) for t in texts]
