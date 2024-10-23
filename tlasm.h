#pragma once

#define PCTLASM "tlasm"

#include <petscpctypes.h>
#include <petscsystypes.h>
#include <petscvec.h>

PetscErrorCode PCTLASM_Create(PC);
PetscErrorCode PCTLASMAddCoarseTemplateVector(PC, Vec);
