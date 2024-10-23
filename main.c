#include <petscdm.h>
#include <petscdmplex.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>

#include "fem.h"
#include "tlasm.h"

static PetscErrorCode CreateMeshFromFilename(MPI_Comm comm, const char *filename, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateGmshFromFile(comm, filename, PETSC_TRUE, dm));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM        dm;
  Mat       A;
  PetscBool flag;
  char      filename[256];
  PetscInt  faces[2] = {5, 5};
  KSP       ksp;
  Vec       x, b;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PCRegister(PCTLASM, PCTLASM_Create));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_file", filename, 512, &flag));
  if (flag) PetscCall(CreateMeshFromFilename(MPI_COMM_WORLD, filename, &dm));
  else PetscCall(DMPlexCreateBoxMesh(MPI_COMM_WORLD, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_FALSE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(AssembleMat(dm, &A, &b));
  PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "solution"));
  PetscCall(VecSet(b, 5.));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dm));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
}
