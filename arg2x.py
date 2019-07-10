# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE: arg2f,arg2i,arg2a, 引数によってパラメータ値を変更するための関数群
"""
test_arg2x.py を参考にして使用する。

"""

#import sys
import re

def arg2f(x,r,s):
    # transform arguments to float value
    fl = "([+-]?[0-9]+[.]?[0-9]*)" # regular expression of float value
    m=re.findall(r+fl,s)
    if m : x=float(m[0])
    return x

def arg2i(x,r,s):
    # transform arguments to float value
    tmp = "([+-]?[0-9]*)" # regular expression of integer value
    m=re.findall(r+tmp,s)
    if m : x=int(m[0])
    return x

def arg2a(x,r,s):
    #tmp = "(\w*)" # regular expression of a-z,A-Z,0-9,and _
    tmp = "([a-zA-Z0-9_\.\/]*)"
    m=re.findall(r+tmp,s)
    if m : x=m[0]
    return x
