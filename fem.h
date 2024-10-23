#pragma once

#include <petscdmplex.h>
#include <petscsys.h>
#include <petscerror.h>
#include <petscsnes.h>
#include <petscfe.h>
#include <petscds.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static void f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -5.;
}

static void f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

#pragma GCC diagnostic pop

PetscErrorCode AssembleMat(DM dm, Mat *A, Vec *b)
{
  SNES           snes;
  PetscInt       dim;
  PetscBool      simplex;
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  DM             cdm;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(SNESCreate(MPI_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetLagPreconditioner(snes, -1));
  PetscCall(SNESSetLagJacobian(snes, -2));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateLagrange(MPI_COMM_WORLD, dim, 1, simplex, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3));

  PetscCall(DMCreateLabel(dm, "boundary"));
  PetscCall(DMGetLabel(dm, "boundary", &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, label));

  PetscCall(DMPlexLabelComplete(dm, label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));

  cdm = dm;
  while (cdm) {
    PetscCall(DMCreateLabel(cdm, "boundary"));
    PetscCall(DMGetLabel(cdm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(cdm, PETSC_DETERMINE, label));
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }

  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetUp(snes));
  PetscCall(DMCreateMatrix(dm, A));
  PetscCall(MatCreateVecs(*A, b, NULL));
  PetscCall(SNESComputeJacobian(snes, *b, *A, *A));

  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}
