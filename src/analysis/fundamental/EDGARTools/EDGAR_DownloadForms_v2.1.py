#!/un3
sr/bin/pytho"""
    Program to download EDGAR files by form type
    ND-SRAF / McDonald : 201606
    https://sraf.nd.edu
    Dependencies (i.e., modules you must already have downloaded)
      EDGAR_Forms.py
      EDGAR_Pac.py
      General_Utilities.py
"""

import os
import time
import sys
# Modify the following statement to identify the path for local modules
sys.path.append('D:\GD\Python\TextualAnalysis\Modules')
# Since these imports are dynamically mapped your IDE might flag an error...it's OK
import EDGAR_Forms  # This module contains some predefined form groups
import EDGAR_Pac
import General_Utilities


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * +

#  NOTES
#        The EDGAR archive contains millions of forms.
#        For details on accessing the EDGAR servers see:
#          https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
#        From that site:
#            "To preserve equitable server access, we ask that bulk FTP
#             transfer requests be performed between 9 PM and 6 AM Eastern 
#             time. Please use efficient scripting, downloading only what you
#             need and space out requests to minimize server load."
#        Note that the program will check the clock every 10 minutes and only
#            download files during the appropriate time.
#        Be a good citizen...keep your requests targeted.
#
#        For large downloads you will sometimes get a hiccup in the server
#            and the file request will fail.  These errs are documented in
#            the log file.  You can manually download those files that fail.
#            Although I attempt to work around server errors, if the SEC's server
#            is sufficiently busy, you might have to try another day.
#
#       For a list of form types and counts by year:
#         "All SEC EDGAR Filings by Type and Year"
#          at https://sraf.nd.edu/textual-analysis/resources/#All%20SEC%20EDGAR%20Filings%20by%20Type%20and%20Year


# -----------------------
# User defined parameters
# -----------------------

# List target forms as strings separated by commas (case sensitive) or
#   load from EDGAR_Forms.  (See EDGAR_Forms module for predefined lists.)
PARM_FORMS = EDGAR_Forms.f_10X  # or, for example, PARM_FORMS = ['8-K', '8-K/A']
PARM_BGNYEAR = 1997  # User selected bgn period.  Earliest available is 1994
PARM_ENDYEAR = 2000  # User selected end period.
PARM_BGNQTR = 1  # Beginning quarter of each year
PARM_ENDQTR = 4  # Ending quarter of each year
# Path where you will store the downloaded files
PARM_PATH = r'C:\EDGAR\FORM_XX\\'
# Change the file pointer below to reflect your location for the log file
#    (directory must already exist)
PARM_LOGFILE = (r'C:\EDGAR\Log_Files\\' +
                r'EDGAR_Download_FORM-X_LogFile_' +
                str(PARM_BGNYEAR) + '-' + str(PARM_ENDYEAR) + '.txt')
# EDGAR parameter
PARM_EDGARPREFIX = 'https://www.sec.gov/Archives/'


#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * +


def download_forms():

    # Download each year/quarter master.idx and save record for requested forms
    f_log = open(PARM_LOGFILE, 'a')
    f_log.write('BEGIN LOOPS:  {0}\n'.format(time.strftime('%c')))
    n_tot = 0
    n_errs = 0
    for year in range(PARM_BGNYEAR, PARM_ENDYEAR + 1):
        for qtr in range(PARM_BGNQTR, PARM_ENDQTR + 1):
            startloop = time.clock()
            n_qtr = 0
            file_count = {}
            # Setup output path
            path = '{0}{1}\\QTR{2}\\'.format(PARM_PATH, str(year), str(qtr))
            if not os.path.exists(path):
                os.makedirs(path)
                print('Path: {0} created'.format(path))
            masterindex = EDGAR_Pac.download_masterindex(year, qtr, True)
            if masterindex:
                for item in masterindex:
                    while EDGAR_Pac.edgar_server_not_available(True):  # kill time when server not available
                        pass
                    if item.form in PARM_FORMS:
                        n_qtr += 1
                        # Keep track of filings and identify duplicates
                        fid = str(item.cik) + str(item.filingdate) + item.form
                        if fid in file_count:
                            file_count[fid] += 1
                        else:
                            file_count[fid] = 1
                        # Setup EDGAR URL and output file name
                        url = PARM_EDGARPREFIX + item.path
                        fname = (path + str(item.filingdate) + '_' + item.form.replace('/', '-') + '_' +
                                 item.path.replace('/', '_'))
                        fname = fname.replace('.txt', '_' + str(file_count[fid]) + '.txt')
                        return_url = General_Utilities.download_to_file(url, fname, f_log)
                        if return_url:
                            n_errs += 1
                        n_tot += 1
                        # time.sleep(1)  # Space out requests
            print(str(year) + ':' + str(qtr) + ' -> {0:,}'.format(n_qtr) + ' downloads completed.  Time = ' +
                  time.strftime('%H:%M:%S', time.gmtime(time.clock() - startloop)) +
                  ' | ' + time.strftime('%c'))
            f_log.write('{0} | {1} | n_qtr = {2:>8,} | n_tot = {3:>8,} | n_err = {4:>6,} | {5}\n'.
                        format(year, qtr, n_qtr, n_tot, n_errs, time.strftime('%c')))

            f_log.flush()

    print('{0:,} total forms downloaded.'.format(n_tot))
    f_log.write('\n{0:,} total forms downloaded.'.format(n_tot))


if __name__ == '__main__':
    start = time.clock()
    print('\n' + time.strftime('%c') + '\nND_SRAF:  Program EDGAR_DownloadForms.py\n')
    download_forms()
    print('\nEDGAR_DownloadForms.py | Normal termination | ' +
          time.strftime('%H:%M:%S', time.gmtime(time.clock() - start)))
    print(time.strftime('%c'))
