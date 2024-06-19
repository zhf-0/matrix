#include "stdio.h"
#include "stdlib.h"
#include "petsc.h"
#include "mpi.h"

static char help[] = "read matrix and vector binary files from file";

int main(int argc, char *argv[])
{
    Mat             A;                  // Pesct Mat
    PetscInt        its;                // Petsc Mat
    Vec             x, b;               // approx solution, RHS
    KSP             ksp;                // linear solver context
    PC              pc;                 // preconditioner
    PetscErrorCode  ierr;
    PetscMPIInt     size, rank;
    PetscViewer     view;                     /* viewer */
    char            mat_file[PETSC_MAX_PATH_LEN];           /* input file name */
    char            yaml_file[PETSC_MAX_PATH_LEN];          
    char            ksp_name[PETSC_MAX_PATH_LEN];          
    char            pc_name[PETSC_MAX_PATH_LEN];          
    char            solve_label[PETSC_MAX_PATH_LEN];          
	int             arg_index = 0;
	FILE            *fp;
    

    ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

	memset(yaml_file,'\0',sizeof(PETSC_MAX_PATH_LEN*sizeof(char)));
	memset(solve_label,'\0',sizeof(PETSC_MAX_PATH_LEN*sizeof(char)));

	while (arg_index < argc)
	{
	  if ( strcmp(argv[arg_index], "-mat_file") == 0 )
	  {
		 arg_index ++;
		 strcpy(mat_file,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-yaml_file") == 0 )
	  {
		 arg_index ++;
		 strcpy(yaml_file,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-ksp_type") == 0 )
	  {
		 arg_index ++;
		 strcpy(ksp_name,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-pc_type") == 0 )
	  {
		 arg_index ++;
		 strcpy(pc_name,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-solve_label") == 0 )
	  {
		 arg_index ++;
		 strcpy(solve_label,argv[arg_index++]);
	  }
	  else
	  {
		  arg_index ++;
	  }
	}


	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat_file,FILE_MODE_READ,&view);CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
	if(size == 1)
	{
		ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
	}
	else
	{
		ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
	}
	ierr = MatLoad(A,view);CHKERRQ(ierr);

	/* MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); */


    // create Vec
    // b
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecSetType(b, VECMPI);
	ierr = VecLoad(b,view);CHKERRQ(ierr);
    

    VecDuplicate(b, &x);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1.e-3,PETSC_DEFAULT,PETSC_DEFAULT,1000);CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    double begin = MPI_Wtime(); 
    double real_begin;
    MPI_Reduce(&begin,&real_begin,1,MPI_DOUBLE,MPI_MIN,0,PETSC_COMM_WORLD);
    
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    double stop = MPI_Wtime();
    double real_stop;
    MPI_Reduce(&stop,&real_stop,1,MPI_DOUBLE,MPI_MAX,0,PETSC_COMM_WORLD);

    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
	KSPGetSolution(ksp,&x);
	KSPGetRhs(ksp,&b);
	Vec r;
    VecDuplicate(b, &r);
	MatMult(A,x,r);
	VecAXPY(r,-1.0,b);
	double res0,res1;
	VecNorm(r,NORM_2,&res0);
	VecNorm(b,NORM_2,&res1);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        write results to yaml file
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	if(rank == 0)
	{
		if(strlen(yaml_file) != 0)
		{
			PetscFOpen(PETSC_COMM_SELF, yaml_file, "a", &fp);

			PetscFPrintf(PETSC_COMM_SELF, fp, "--- \n");

			if(strlen(solve_label) != 0)
				PetscFPrintf(PETSC_COMM_SELF, fp, "solve_label: %s \n",solve_label);

			PetscFPrintf(PETSC_COMM_SELF, fp, "iter: %d \n",its);
			PetscFPrintf(PETSC_COMM_SELF, fp, "stop_reason: %d \n",reason);
			PetscFPrintf(PETSC_COMM_SELF, fp, "r_norm: %e \n",res0);
			PetscFPrintf(PETSC_COMM_SELF, fp, "b_norm: %e \n",res1);
			PetscFPrintf(PETSC_COMM_SELF, fp, "relative_norm: %e \n",res0/res1);
			PetscFPrintf(PETSC_COMM_SELF, fp, "time: %e \n",real_stop-real_begin);
			PetscFPrintf(PETSC_COMM_SELF, fp, "processed: 0 \n");

			PetscFClose(PETSC_COMM_SELF, fp);
		}


		ierr = PetscPrintf(PETSC_COMM_SELF,"================================= \n");CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"process matrix %s using %s \n",mat_file,solve_label);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"iteration_num = %d \n",its);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"converge_reason = %d \n",reason);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"residula_norm = %e \n",res0);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"b_norm = %e \n",res1);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"relative_res = %e \n",res0/res1);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_SELF,"elapsed_time = %e \n",real_stop-real_begin);CHKERRQ(ierr);
	}

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              free all resource
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	PetscViewerDestroy(&view);
    ierr = VecDestroy(&x);CHKERRQ(ierr); 
    ierr = VecDestroy(&b);CHKERRQ(ierr); 
    ierr = VecDestroy(&r);CHKERRQ(ierr); 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

    ierr = PetscFinalize();

    return ierr;
}
