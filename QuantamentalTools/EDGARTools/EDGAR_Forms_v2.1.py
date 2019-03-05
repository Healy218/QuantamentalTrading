#!/usr/bin/python3
"""
    Predefined SEC form strings
    Feel free to add your own lists
    ND-SRAF / McDonald : 201606
"""

f_10K = ['10-K', '10-K405', '10KSB', '10-KSB', '10KSB40']
f_10KA = ['10-K/A', '10-K405/A', '10KSB/A', '10-KSB/A', '10KSB40/A']
f_10KT = ['10-KT', '10KT405', '10-KT/A', '10KT405/A']
f_10Q = ['10-Q', '10QSB', '10-QSB']
f_10QA = ['10-Q/A', '10QSB/A', '10-QSB/A']
f_10QT = ['10-QT', '10-QT/A']
# List of all 10-X related forms
f_10X = f_10K + f_10KA + f_10KT + f_10Q + f_10QA + f_10QT
# Regulation A+ related forms
f_1X = ['1-A', '1-A/A', '1-K', '1-SA', '1-U', '1-Z']
