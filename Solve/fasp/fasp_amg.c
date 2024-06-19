/*! \file  poisson-amg.c
 *
 *  \brief The first test example for FASP: using AMG to solve 
 *         the discrete Poisson equation from P1 finite element.
 *         C version.
 *
 *  \note  Solving the Poisson equation (P1 FEM) with AMG: C version
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2011--2018 by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

typedef struct
{
    INT iter;
    REAL time;
    REAL r_norm;
    REAL b_norm;
    REAL relative_norm;
    
} Res; 

void write_yaml(const char *filename,
				const char *label, 
				Res * result)
{
    FILE *fp = fopen(filename, "a");

    if ( fp == NULL ) fasp_chkerr(ERROR_OPEN_FILE, filename);

	fprintf(fp, "---\n");
	if (strlen(label) != 0) fprintf(fp, "solve_label: %s \n", label);
	fprintf(fp, "iter: %d \n", result->iter);
	fprintf(fp, "time: %e \n", result->time);
	fprintf(fp, "r_norm: %e \n", result->r_norm);
	fprintf(fp, "b_norm: %e \n", result->b_norm);
	fprintf(fp, "relative_norm: %e \n", result->relative_norm);
	fprintf(fp, "processed: 0 \n");

    fclose(fp);
}


INT local_fasp_solver_dcsr_krylov_amg (dCSRmat    *A,
									   dvector    *b,
									   dvector    *x,
									   ITS_param  *itparam,
									   AMG_param  *amgparam,
									   Res        *result)
{
    const SHORT prtlvl     = itparam->print_level;
    const SHORT max_levels = amgparam->max_levels;
    const INT   nnz = A->nnz, m = A->row, n = A->col;

    /* Local Variables */
    INT  status = FASP_SUCCESS;
    REAL solve_start, solve_end;

#if MULTI_COLOR_ORDER
    A->color = 0;
    A->IC    = NULL;
    A->ICMAP = NULL;
#endif

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
    printf("### DEBUG: matrix size: %d %d %d\n", A->row, A->col, A->nnz);
    printf("### DEBUG: rhs/sol size: %d %d\n", b->row, x->row);
#endif

    fasp_gettime(&solve_start);

    // initialize A, b, x for mgl[0]
    AMG_data* mgl = fasp_amg_data_create(max_levels);
    mgl[0].A      = fasp_dcsr_create(m, n, nnz);
    fasp_dcsr_cp(A, &mgl[0].A);
    mgl[0].b = fasp_dvec_create(n);
    mgl[0].x = fasp_dvec_create(n);

    // setup preconditioner
    switch (amgparam->AMG_type) {

        case SA_AMG: // Smoothed Aggregation AMG
            status = fasp_amg_setup_sa(mgl, amgparam);
            break;

        case UA_AMG: // Unsmoothed Aggregation AMG
            status = fasp_amg_setup_ua(mgl, amgparam);
            break;

        default: // Classical AMG
            status = fasp_amg_setup_rs(mgl, amgparam);
    }

#if DEBUG_MODE > 1
    fasp_mem_usage();
#endif

    if (status < 0) goto FINISHED;

    // setup preconditioner
    precond_data pcdata;
    fasp_param_amg_to_prec(&pcdata, amgparam);
    pcdata.max_levels = mgl[0].num_levels;
    pcdata.mgl_data   = mgl;

    precond pc;
    pc.data = &pcdata;

    if (itparam->precond_type == PREC_FMG) {
        pc.fct = fasp_precond_famg; // Full AMG
    } else {
        switch (amgparam->cycle_type) {
            case AMLI_CYCLE: // AMLI cycle
                pc.fct = fasp_precond_amli;
                break;
            case NL_AMLI_CYCLE: // Nonlinear AMLI
                pc.fct = fasp_precond_namli;
                break;
            default: // V,W-cycles or hybrid cycles
                pc.fct = fasp_precond_amg;
        }
    }

    // call iterative solver
    status = fasp_solver_dcsr_itsolver(A, b, x, &pc, itparam);

	fasp_gettime(&solve_end);
    if (prtlvl >= PRINT_MIN) {
        fasp_cputime("AMG_Krylov method totally", solve_end - solve_start);
    }


	// collect results
	REAL b_norm = fasp_blas_dvec_norm2(b);

	dvector resi;
    fasp_dvec_alloc(m, &resi);
    fasp_dvec_cp(b, &resi);
    fasp_blas_dcsr_aAxpy(-1.0, A, x->val, resi.val);
	REAL r_norm = fasp_blas_dvec_norm2(&resi); 
    fasp_dvec_free(&resi);

	result->time = solve_end - solve_start;
	result->iter = status;
	result->b_norm = b_norm;
	result->r_norm = r_norm;
	result->relative_norm = r_norm/b_norm;
    
FINISHED:
    fasp_amg_data_free(mgl, amgparam);
    
#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
    
    return status;
}

int main (int argc, const char * argv[]) 
{
	/* char * mat_file = NULL; */
	/* char * vec_file = NULL; */
	char mat_file[512];
	char vec_file[512];
	char yaml_file[512];
	char solve_label[512];
	REAL couple = -1.0;
	memset(yaml_file,'\0',sizeof(512*sizeof(char)));
	memset(solve_label,'\0',sizeof(512*sizeof(char)));
    
	int arg_index = 0;
	while (arg_index < argc)
	{
	  if ( strcmp(argv[arg_index], "-mat_file") == 0 )
	  {
			 arg_index ++;
			 strcpy(mat_file,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-vec_file") == 0 )
	  {
			 arg_index ++;
			 strcpy(vec_file,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-yaml_file") == 0 )
	  {
			 arg_index ++;
			 strcpy(yaml_file,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-solve_label") == 0 )
	  {
			 arg_index ++;
			 strcpy(solve_label,argv[arg_index++]);
	  }
	  else if ( strcmp(argv[arg_index], "-couple") == 0 )
	  {
					 arg_index ++;
					 couple = atof(argv[arg_index++]);
	  }
	  else
	  {
			  arg_index ++;
	  }
	}

    input_param     inparam;  // parameters from input files
    ITS_param       itspar; // parameters for itsolver
    AMG_param       amgparam; // parameters for AMG

    fasp_param_set(argc, argv, &inparam);

    if (couple > 0.0 && couple < 1.0)
    {
        inparam.AMG_strong_coupled = couple;
    }

    fasp_param_init(&inparam, &itspar, &amgparam, NULL, NULL);



    // Set local parameters using the input values
    const int print_level = inparam.print_level;
    
    // Step 1. Get stiffness matrix and right-hand side
    // Read A and b -- P1 FE discretization for Poisson. The location
    // of the data files is given in "ini/amg.dat".
    dCSRmat A;
    dvector b, x;
    
    /* fasp_dcoo_read(mat_file, &A); */
    fasp_dcoo_read1(mat_file, &A);
    fasp_dvec_read(vec_file, &b);

    // Step 2. Print problem size and AMG parameters
    if (print_level>PRINT_NONE) {
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        fasp_param_amg_print(&amgparam);
    }
    
    // Step 3. Solve the system with AMG as an iterative solver
    // Set the initial guess to be zero and then solve it
    // with AMG method as an iterative procedure
    fasp_dvec_alloc(A.row, &x);
    fasp_dvec_set(A.row, &x, 0.0);
    
    if (print_level>PRINT_NONE) {
		fasp_param_solver_print(&itspar);
		fasp_param_amg_print(&amgparam);
    }

    //fasp_solver_amg(&A, &b, &x, &amgparam);
	Res result;
    local_fasp_solver_dcsr_krylov_amg(&A, &b, &x, &itspar, &amgparam, &result);
	printf("iter = %d, time = %e \n", result.iter, result.time);

	if (strlen(yaml_file) != 0)
		write_yaml(yaml_file,solve_label,&result);
    
    // Step 4. Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&x);
    
    return FASP_SUCCESS;
}


