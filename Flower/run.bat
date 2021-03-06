@ECHO OFF   
if [%2]==[] goto error
if [%1]==[] if [%2]==[] goto error

set id_file=_clients.txt
set id_file=%1%id_file%
echo %id_file%

set id_folder=../data/Flower/%2/%id_file%
echo %id_folder%

START "server" CMD /k python.exe Flower_Server.py  
for /f "tokens=*" %%s in (%id_folder%) do (
  echo client %%s
  START "client" CMD /k python.exe %%s %2 "8080" 
)
goto :eof

:error
echo Parameters missing. [1]: number of clients [2]: folder of id_file
exit /B
