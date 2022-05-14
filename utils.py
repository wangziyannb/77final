# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 1:00
# @Author  : Wang Ziyan
# @Email   : 1269586767@qq.com
# @File    : utils.py
# @Software: PyCharm
import re


def flatten(a):
    if not isinstance(a, (list,)):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


def preprocessing(source):
    if source == '\n' or source == '':
        return []
    source = str(source).strip('\n').strip(' ')
    source = source.lower()
    source = re.sub(r'[^a-z]+', ' ', source)
    source = source.strip(' ')
    source = source.split(' ')
    return source
