#!/bin/bash
echo Starting...
echo Baseline V1
python3 uncertainty.py -m mistral -g 0
echo FIP1 V1
python3 uncertainty.py -m mistral -f -g 0
echo FIP2 V1
python3 uncertainty.py -m mistral -f -fn 2 -g 0
echo FIP5 V1
python3 uncertainty.py -m mistral -f -fn 5 -g 0
echo FIP10 V1
python3 uncertainty.py -m mistral -f -fn 10 -g 0
echo RIP V1
python3 uncertainty.py -m mistral -r -g 0
echo Baseline V2
python3 uncertainty.py -m mistral -v 2 -g 0
echo FIP1 V2
python3 uncertainty.py -m mistral -f -v 2 -g 0
echo FIP2 V2
python3 uncertainty.py -m mistral -f -fn 2 -v 2 -g 0
echo FIP3 V2
python3 uncertainty.py -m mistral -f -fn 5 -v 2 -g 0
echo FIP5 V2
python3 uncertainty.py -m mistral -f -fn 10 -v 2 -g 0
echo FIP10 V2
python3 uncertainty.py -m mistral -r -v 2 -g 0
echo DONE!
