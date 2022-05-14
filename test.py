# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 8:31
# @Author  : Wang Ziyan
# @Email   : 1269586767@qq.com
# @File    : test.py
# @Software: PyCharm
import re


def preprocessing(source):
    if source == '\n' or source == '':
        return []
    source = str(source).strip('\n').strip(' ')
    source = source.lower()
    source = re.sub(r'[^a-z]+', ' ', source)
    source = source.strip(' ')
    source = source.split(' ')
    return source


def load_material(textpath):
    ls = []
    with open(textpath, 'r', encoding="utf-8") as o:
        for i in o.readlines():
            l = preprocessing(i)
            if len(l) == 0:
                continue
            ls.append(l)
    return ls

if __name__ == '__main__':
    a=load_material('74-0.txt')
    print()