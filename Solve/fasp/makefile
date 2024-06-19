CC = gcc
# FLAG = -O3
FLAG = -g
inc = -I ../install/include
lib = -L ../install/lib -lfasp -lm


fasp:fasp_amg.c AuxParam.c
	${CC} ${FLAG} ${inc} $^ ${lib} -o $@
