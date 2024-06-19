mpipath = /home/zhf/software/openmpi/install
petscpath = /home/zhf/software/petsc/install

CC = ${mpipath}/bin/mpicc
EXE = ${mpipath}/bin/mpiexec
CFLAGS = 
np = 2
opt = 1

ifeq ($(opt),0)
	incpath = -I ${petscpath}/include -I ${mpipath}/include 
	libpath = -Wl,-rpath ${petscpath}/lib -L ${petscpath}/lib -Wl,-rpath ${mpipath}/lib -L ${mpipath}/lib  
	CFLAGS += -Wall -g
else
	incpath = -I ${petscpath}/include -I ${mpipath}/include
	libpath = -Wl,-rpath ${petscpath}/lib -L ${petscpath}/lib -Wl,-rpath ${mpipath}/lib -L ${mpipath}/lib 
	CFLAGS += -O3
endif

all: rs

rs:ReadAndSol.c 
	${CC} ${CFLAGS}  ${incpath} $^ ${libpath} -lpetsc -lmpi -o $@  


.PHONY : clean

clean:
	rm *.o rs  

   
