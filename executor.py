import os
import glob
import time

while True:
    files = glob.glob('logs/*.sh')
    if len(files) == 0:
        print('No target, executor idle ...                         ', end='\r')
        time.sleep(1)
        continue
    files.sort()
    print(f'Executing {files[0]} ...                                ')
    print(f'{len(files)-1} jobs waiting.                            ', end='\r')
    os.system(f'sh {files[0]}')
    os.system(f'mv {files[0]} {files[0]}.old')
