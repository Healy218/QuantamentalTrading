#!/usr/bin/python3
"""
    General utility programs
    ND-SRAF / McDonald : 201606
    https://sraf.nd.edu
"""


import time
from urllib.request import urlretrieve
from urllib.request import urlopen


def download_to_file(_url, _fname, _f_log=None):
    # download file from 'url' and write to '_fname'
    # Loop accounts for temporary server/ISP issues

    number_of_tries = 10
    sleep_time = 10  # Note sleep time accumulates according to err

    for i in range(1, number_of_tries + 1):
        try:
            urlretrieve(_url, _fname)
            return
        except Exception as exc:
            if i == 1:
                print('\n==>urlretrieve error in download_to_file.py')
            print('  {0}. _url:  {1}'.format(i, _url))
            print('     _fname:  {0}'.format(_fname))
            print('     Warning: {0}  [{1}]'.format(str(exc), time.strftime('%c')))
            if '404' in str(exc):
                break
            print('     Retry in {0} seconds'.format(sleep_time))
            time.sleep(sleep_time)
            sleep_time += sleep_time

    print('\n  ERROR:  Download failed for')
    print('          url:  {0}'.format(_url))
    print('          _fname:  {0}'.format(_fname))
    if _f_log:
        _f_log.write('ERROR:  Download failed=>')
        _f_log.write('  _url: {0:75}'.format(_url))
        _f_log.write('  |  _fname: {0}'.format(_fname))
        _f_log.write('  |  {0}\n'.format(time.strftime('%c')))

    return


def download_to_doc(_url, _f_log=None):
    # Download url content to string doc
    # Loop accounts for temporary server/ISP issues

    number_of_tries = 10
    sleep_time = 5  # Note sleep time is cumulative over loop

    for i in range(1, number_of_tries + 1):
        try:
            doc = urlopen(_url).read().decode('utf-8', errors='ignore')
            return
        except Exception as exc:
            if i == 1:
                print('\n==>urlopen error in download_to_doc.py')
            print('  {0}. _url:  {1}'.format(i, _url))
            print('     Warning: {0}  [{1}]'.format(str(exc), time.strftime('%c')))
            if '404' in str(exc):
                break
            print('     Retry in {0} seconds'.format(sleep_time))
            time.sleep(sleep_time)
            sleep_time += sleep_time

    print('\n  ERROR:  Download failed for url: {0}'.format(_url))
    if _f_log:
        _f_log.write('ERROR:  Download failed=>  _url: {0:75}'.format(_url))
        _f_log.write('  |  {0}\n'.format(time.strftime('%c')))

    return None


# Test routine
if __name__ == '__main__':
    # Note:  This test is setup to throw errors
    print('\n' + time.strftime('%c') + '\nND_SRAF:  Program General_Utilities.py\n')
    test_url = 'http://www.nd.edu/~mcdonald/xyz.html'  # set to throw an error
    fname = 'D:/Temp/DL_test.txt'
    f_log = open('D:/Temp/DL_log.txt', 'w')
    download_to_file(test_url, fname, f_log)
    doc_url = 'http://www.sec.gov/Archives/edgar/data/1046568/0001193125-15-075170.zzz'
    doc = download_to_doc(doc_url, f_log)
    print('\nNormal termination.')
    print(time.strftime('%c'))
