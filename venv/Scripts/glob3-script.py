#!C:\Users\vasilyev\Documents\PyCharm\self-focusing_3d\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'glob3==0.0.1','console_scripts','glob3'
__requires__ = 'glob3==0.0.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('glob3==0.0.1', 'console_scripts', 'glob3')()
    )
