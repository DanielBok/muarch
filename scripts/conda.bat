@ECHO OFF

ECHO Remember to run this script from the muarch root folder
conda deactivate

IF %1==3.6 GOTO env36
IF %1==3.7 GOTO env37

ECHO Unrecognized version "%1". Use 3.6 or 3.7
EXIT /B

:env36
ECHO Running python 3.6
conda activate b36
GOTO build

:env37
ECHO Running python 3.7
START /B /WAIT conda activate b37
GOTO build

:build
ECHO Building package
conda build --output-folder dist conda.recipe

conda build purge
conda deactivate
