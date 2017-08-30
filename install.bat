
python setup.py install

chdir doc
call make.bat html
chdir ..

py.test

pause
exit
