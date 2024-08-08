#!/bin/bash
echo Starting...
echo Baseline V1
python3 uncertainty.py -m gpt-4o -g 0
echo FIP1 V1
python3 uncertainty.py -m gpt-4o -f -g 0
echo FIP2 V1
python3 uncertainty.py -m gpt-4o -f -fn 2 -g 0
echo FIP5 V1
python3 uncertainty.py -m gpt-4o -f -fn 5 -g 0
echo FIP10 V1
python3 uncertainty.py -m gpt-4o -f -fn 10 -g 0
echo RIP V1
python3 uncertainty.py -m gpt-4o -r -g 0
echo Baseline V2
python3 uncertainty.py -m gpt-4o -v 2 -g 0
echo FIP1 V2
python3 uncertainty.py -m gpt-4o -f -v 2 -g 0
echo FIP2 V2
python3 uncertainty.py -m gpt-4o -f -fn 2 -v 2 -g 0
echo FIP3 V2
python3 uncertainty.py -m gpt-4o -f -fn 5 -v 2 -g 0
echo FIP5 V2
python3 uncertainty.py -m gpt-4o -f -fn 10 -v 2 -g 0
echo FIP10 V2
python3 uncertainty.py -m gpt-4o -r -v 2 -g 0
echo DONE!