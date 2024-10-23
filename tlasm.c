#include "tlasm.h"

#include <mpi.h>
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/pcasmimpl.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <time.h>

typedef struct {
  PC        comp, osm, cpc; // The composite PC, the additive Schwarz PC, and the coarse PC (which is typically of type PCGALERKIN)
  Mat       Z;
  PetscBool with_coarse;
} *PC_TLASM;

static PetscErrorCode PCDestroy_TLASM(PC pc)
{
  PC_TLASM tlasm = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCDestroy(&tlasm->osm));
  PetscCall(PCDestroy(&tlasm->comp));
  PetscCall(PCDestroy(&tlasm->cpc));
  PetscCall(PetscFree(tlasm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTLASMAddCoarseTemplateVector(PC pc, Vec vec)
{
  PetscFunctionBeginUser;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TLASMSetUpNicolaides(PC pc)
{
  PC_TLASM    tlasm = pc->data;
  PC_ASM     *osm   = tlasm->osm->data;
  Vec         ones;
  PetscMPIInt rank;
  PetscInt    m, n;
  Mat         Z, C;
  Vec         z;
  KSP         ksp;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
  PetscCall(MatGetSize(pc->pmat, &m, NULL));
  PetscCall(MatGetLocalSize(pc->pmat, &n, NULL));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)pc), n, 1, m, osm->n, NULL, &Z));
  PetscCall(VecDuplicate(osm->lx, &ones));
  for (PetscInt i = 0; i < osm->n; ++i) {
    if (rank == i) {
      PetscCall(VecSet(ones, 1.));
      /* PetscCall(VecPointwiseMult(ones, tlasm->D, ones)); */
    } else PetscCall(VecSet(ones, 0.));
    PetscCall(MatDenseGetColumnVecWrite(Z, i, &z));
    PetscCall(VecSet(z, 0.));
    PetscCall(VecScatterBegin(osm->restriction, ones, z, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(osm->restriction, ones, z, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(MatDenseRestoreColumnVecWrite(Z, i, &z));
  }
  PetscCall(VecDestroy(&ones));
  PetscCall(MatViewFromOptions(Z, NULL, "-view_Z_mat"));
  PetscCall(MatPtAP(pc->pmat, Z, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatViewFromOptions(C, NULL, "-view_coarse_mat"));
  PetscCall(VecDestroy(&ones));
  tlasm->Z = Z;
  PetscCall(PCGalerkinSetInterpolation(tlasm->cpc, tlasm->Z));
  PetscCall(PCGalerkinGetKSP(tlasm->cpc, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_TLASM(PC pc)
{
  PC_TLASM  tlasm = pc->data;
  PC_ASM   *osm   = tlasm->osm->data;
  PCType    type;
  PetscBool flag = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(tlasm->comp, PCCOMPOSITE));
  PetscCall(PCSetOperators(tlasm->comp, pc->pmat, pc->pmat));
  PetscCall(PCSetUp(tlasm->comp));

  PetscCall(PCSetType(tlasm->osm, PCASM));
  PetscCall(PCSetOperators(tlasm->osm, pc->pmat, pc->pmat));
  PetscCall(PCSetUp(tlasm->osm));
  PetscCheck(osm->n_local == osm->n_local_true && osm->n_local == 1, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only one subdomain per process supported");

  PetscCall(PCSetOperators(tlasm->cpc, pc->pmat, pc->pmat));
  PetscCall(PCGetType(tlasm->cpc, &type));
  PetscCall(PetscStrcmp(type, PCNONE, &flag));
  if (!flag) PetscCall(TLASMSetUpNicolaides(pc));
  PetscCall(PCSetUp(tlasm->cpc));

  PetscCall(PCCompositeAddPC(tlasm->comp, tlasm->cpc));
  PetscCall(PCCompositeAddPC(tlasm->comp, tlasm->osm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_TLASM(PC pc, Vec x, Vec y)
{
  PC_TLASM tlasm = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCApply(tlasm->comp, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_TLASM(PC pc, Vec x, Vec y)
{
  PC_TLASM tlasm = pc->data;

  PetscFunctionBeginUser;
  PetscCall(PCApplyTranspose(tlasm->comp, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTLASMSetOverlap(PC pc, PetscInt overlap)
{
  PC_TLASM tlasm = pc->data;

  PetscFunctionBeginUser;
  PetscCheck(overlap >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Overlap must be greater or equal than zero");
  PetscCall(PCTLASMSetOverlap(tlasm->osm, overlap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_TLASM(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_TLASM  tlasm = pc->data;
  PetscInt  ovl;
  PetscBool flag;

  PetscFunctionBeginUser;
  PetscOptionsHeadBegin(PetscOptionsObject, "Two-level additive Schwarz options");
  PetscCall(PetscOptionsInt("-pc_tlasm_overlap", "Number of grid points overlap", NULL, ((PC_ASM *)tlasm->osm->data)->overlap, &ovl, &flag));
  if (flag) PetscCall(PCTLASMSetOverlap(pc, ovl));
  PetscCall(PetscOptionsBool("-pc_tlasm_with_coarse", "Perform a coarse solve", NULL, tlasm->with_coarse, &tlasm->with_coarse, NULL));
  PetscCall(PCSetFromOptions(tlasm->comp));
  PetscCall(PCSetFromOptions(tlasm->osm));
  PetscCall(PCSetFromOptions(tlasm->cpc));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTLASM_Create(PC pc)
{
  PC_TLASM tlasm;
  MPI_Comm comm = PetscObjectComm((PetscObject)pc);

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&tlasm));
  tlasm->with_coarse      = PETSC_FALSE;
  pc->data                = tlasm;
  pc->ops->setup          = PCSetUp_TLASM;
  pc->ops->setfromoptions = PCSetFromOptions_TLASM;
  pc->ops->apply          = PCApply_TLASM;
  pc->ops->applytranspose = PCApplyTranspose_TLASM;
  pc->ops->destroy        = PCDestroy_TLASM;

  PetscCall(PCCreate(comm, &tlasm->comp));
  PetscCall(PCSetType(tlasm->comp, PCCOMPOSITE));
  PetscCall(PCSetOptionsPrefix(tlasm->comp, "tlasm_"));

  PetscCall(PCCreate(comm, &tlasm->osm));
  PetscCall(PCSetType(tlasm->osm, PCASM));
  PetscCall(PCSetOptionsPrefix(tlasm->osm, "fine_"));

  PetscCall(PCCreate(comm, &tlasm->cpc));
  PetscCall(PCSetType(tlasm->cpc, PCGALERKIN));
  PetscCall(PCSetOptionsPrefix(tlasm->cpc, "coarse_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
