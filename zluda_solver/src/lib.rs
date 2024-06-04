#![allow(non_snake_case)]
#[allow(warnings)]
mod cusolver;
pub use cusolver::*;

use hipsolver_sys::*;

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> cusolverStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> cusolverStatus_t {
    cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED
}

fn to_cuda(status: hipsolverStatus_t) -> cusolverStatus_t {
    match status {
        hipsolverStatus_t::HIPSOLVER_STATUS_SUCCESS => cusolverStatus_t::CUSOLVER_STATUS_SUCCESS,
        hipsolverStatus_t::HIPSOLVER_STATUS_NOT_INITIALIZED => {
            cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_ALLOC_FAILED => {
            cusolverStatus_t::CUSOLVER_STATUS_ALLOC_FAILED
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_INVALID_VALUE => {
            cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_MAPPING_ERROR => {
            cusolverStatus_t::CUSOLVER_STATUS_MAPPING_ERROR
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_EXECUTION_FAILED => {
            cusolverStatus_t::CUSOLVER_STATUS_EXECUTION_FAILED
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_INTERNAL_ERROR => {
            cusolverStatus_t::CUSOLVER_STATUS_INTERNAL_ERROR
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_NOT_SUPPORTED => {
            cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_ARCH_MISMATCH => {
            cusolverStatus_t::CUSOLVER_STATUS_ARCH_MISMATCH
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_INVALID_ENUM => {
            cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_HANDLE_IS_NULLPTR => {
            cusolverStatus_t::CUSOLVER_STATUS_INVALID_VALUE
        }
        hipsolverStatus_t::HIPSOLVER_STATUS_UNKNOWN => {
            cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED
        }
        _ => panic!(),
    }
}

fn to_fill(mode: cublasFillMode_t) -> hipsolverFillMode_t {
    match mode {
        cublasFillMode_t::CUBLAS_FILL_MODE_UPPER => hipsolverFillMode_t::HIPSOLVER_FILL_MODE_UPPER,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER => hipsolverFillMode_t::HIPSOLVER_FILL_MODE_LOWER,
        _ => panic!(),
    }
}

fn op_from_cuda(op: cublasOperation_t) -> hipsolverOperation_t {
    match op {
        cublasOperation_t::CUBLAS_OP_C => hipsolverOperation_t::HIPSOLVER_OP_C,
        cublasOperation_t::CUBLAS_OP_N => hipsolverOperation_t::HIPSOLVER_OP_N,
        cublasOperation_t::CUBLAS_OP_T => hipsolverOperation_t::HIPSOLVER_OP_T,
        _ => panic!(),
    }
}

fn to_side_mode(mode: cublasSideMode_t) -> hipsolverSideMode_t {
    match mode {
        cublasSideMode_t::CUBLAS_SIDE_LEFT => hipsolverSideMode_t::HIPSOLVER_SIDE_LEFT,
        cublasSideMode_t::CUBLAS_SIDE_RIGHT => hipsolverSideMode_t::HIPSOLVER_SIDE_LEFT,
        _ => panic!(),
    }
}

fn to_eig_mode(mode: cusolverEigMode_t) -> hipsolverEigMode_t {
    match mode {
        cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR => {
            hipsolverEigMode_t::HIPSOLVER_EIG_MODE_NOVECTOR
        }
        cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR => {
            hipsolverEigMode_t::HIPSOLVER_EIG_MODE_VECTOR
        }
        _ => panic!(),
    }
}

fn to_eig_range(range: cusolverEigRange_t) -> hipsolverEigRange_t {
    match range {
        cusolverEigRange_t::CUSOLVER_EIG_RANGE_ALL => hipsolverEigRange_t::HIPSOLVER_EIG_RANGE_ALL,
        cusolverEigRange_t::CUSOLVER_EIG_RANGE_I => hipsolverEigRange_t::HIPSOLVER_EIG_RANGE_I,
        cusolverEigRange_t::CUSOLVER_EIG_RANGE_V => hipsolverEigRange_t::HIPSOLVER_EIG_RANGE_V,
        _ => panic!(),
    }
}

fn to_eig_type(eig_type: cusolverEigType_t) -> hipsolverEigType_t {
    match eig_type {
        cusolverEigType_t::CUSOLVER_EIG_TYPE_1 => hipsolverEigType_t::HIPSOLVER_EIG_TYPE_1,
        cusolverEigType_t::CUSOLVER_EIG_TYPE_2 => hipsolverEigType_t::HIPSOLVER_EIG_TYPE_2,
        cusolverEigType_t::CUSOLVER_EIG_TYPE_3 => hipsolverEigType_t::HIPSOLVER_EIG_TYPE_3,
        _ => panic!(),
    }
}

fn to_rf_matrix(format: cusolverRfMatrixFormat_t) -> hipsolverRfMatrixFormat_t {
    match format {
        cusolverRfMatrixFormat_t::CUSOLVERRF_MATRIX_FORMAT_CSC => {
            hipsolverRfMatrixFormat_t::HIPSOLVERRF_MATRIX_FORMAT_CSC
        }
        cusolverRfMatrixFormat_t::CUSOLVERRF_MATRIX_FORMAT_CSR => {
            hipsolverRfMatrixFormat_t::HIPSOLVERRF_MATRIX_FORMAT_CSR
        }
        _ => panic!(),
    }
}

fn to_rf_dia(diag: cusolverRfUnitDiagonal_t) -> hipsolverRfUnitDiagonal_t {
    match diag {
        cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L => {
            hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
        }
        cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U => {
            hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U
        }
        cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_STORED_L => {
            hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_STORED_L
        }
        cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_STORED_U => {
            hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_STORED_U
        }
        _ => panic!(),
    }
}

fn from_hip_to_rf_matrix(format: hipsolverRfMatrixFormat_t) -> cusolverRfMatrixFormat_t {
    match format {
        hipsolverRfMatrixFormat_t::HIPSOLVERRF_MATRIX_FORMAT_CSC => {
            cusolverRfMatrixFormat_t::CUSOLVERRF_MATRIX_FORMAT_CSC
        }
        hipsolverRfMatrixFormat_t::HIPSOLVERRF_MATRIX_FORMAT_CSR => {
            cusolverRfMatrixFormat_t::CUSOLVERRF_MATRIX_FORMAT_CSR
        }
        _ => panic!(),
    }
}

fn from_hip_to_rf_dia(diag: hipsolverRfUnitDiagonal_t) -> cusolverRfUnitDiagonal_t {
    match diag {
        hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L => {
            cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
        }
        hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U => {
            cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U
        }
        hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_STORED_L => {
            cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_STORED_L
        }
        hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_STORED_U => {
            cusolverRfUnitDiagonal_t::CUSOLVERRF_UNIT_DIAGONAL_STORED_U
        }
        _ => panic!(),
    }
}

// fn to_rf_report(report: cusolverRfNumericBoostReport_t) -> hipsolverRfNumericBoostReport_t {
//     match report {
//         cusolverRfNumericBoostReport_t::CUSOLVERRF_NUMERIC_BOOST_NOT_USED => {
//             hipsolverRfNumericBoostReport_t::HIPSOLVERRF_NUMERIC_BOOST_NOT_USED
//         }
//         cusolverRfNumericBoostReport_t::CUSOLVERRF_NUMERIC_BOOST_USED => {
//             hipsolverRfNumericBoostReport_t::HIPSOLVERRF_NUMERIC_BOOST_USED
//         }
//         _ => panic!(),
//     }
// }

fn from_hip_to_rf_report(
    report: hipsolverRfNumericBoostReport_t,
) -> cusolverRfNumericBoostReport_t {
    match report {
        hipsolverRfNumericBoostReport_t::HIPSOLVERRF_NUMERIC_BOOST_NOT_USED => {
            cusolverRfNumericBoostReport_t::CUSOLVERRF_NUMERIC_BOOST_NOT_USED
        }
        hipsolverRfNumericBoostReport_t::HIPSOLVERRF_NUMERIC_BOOST_USED => {
            cusolverRfNumericBoostReport_t::CUSOLVERRF_NUMERIC_BOOST_USED
        }
        _ => panic!(),
    }
}

fn to_rf_solve(solve: cusolverRfTriangularSolve_t) -> hipsolverRfTriangularSolve_t {
    match solve {
        cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 => {
            hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1
        }
        cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 => {
            hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2
        }
        cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 => {
            hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3
        }
        _ => panic!(),
    }
}

fn from_hip_to_rf_solve(solve: hipsolverRfTriangularSolve_t) -> cusolverRfTriangularSolve_t {
    match solve {
        hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 => {
            cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
        }
        hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2 => {
            cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG2
        }
        hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3 => {
            cusolverRfTriangularSolve_t::CUSOLVERRF_TRIANGULAR_SOLVE_ALG3
        }
        _ => panic!(),
    }
}

fn to_rf_factor(factor: cusolverRfFactorization_t) -> hipsolverRfFactorization_t {
    match factor {
        cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG0 => {
            hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG0
        }
        cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG1 => {
            hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG1
        }
        cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG2 => {
            hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG2
        }
        _ => panic!(),
    }
}

fn from_hip_to_rf_factor(factor: hipsolverRfFactorization_t) -> cusolverRfFactorization_t {
    match factor {
        hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG0 => {
            cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG0
        }
        hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG1 => {
            cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG1
        }
        hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG2 => {
            cusolverRfFactorization_t::CUSOLVERRF_FACTORIZATION_ALG2
        }
        _ => panic!(),
    }
}

fn to_rf_mode(mode: cusolverRfResetValuesFastMode_t) -> hipsolverRfResetValuesFastMode_t {
    match mode {
        cusolverRfResetValuesFastMode_t::CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF => {
            hipsolverRfResetValuesFastMode_t::HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF
        }
        cusolverRfResetValuesFastMode_t::CUSOLVERRF_RESET_VALUES_FAST_MODE_ON => {
            hipsolverRfResetValuesFastMode_t::HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON
        }
        _ => panic!(),
    }
}

fn from_hip_to_rf_mode(mode: hipsolverRfResetValuesFastMode_t) -> cusolverRfResetValuesFastMode_t {
    match mode {
        hipsolverRfResetValuesFastMode_t::HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF => {
            cusolverRfResetValuesFastMode_t::CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF
        }
        hipsolverRfResetValuesFastMode_t::HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON => {
            cusolverRfResetValuesFastMode_t::CUSOLVERRF_RESET_VALUES_FAST_MODE_ON
        }
        _ => panic!(),
    }
}

unsafe fn dn_create(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnCreate(handle.cast()))
}

unsafe fn dn_destroy(handle: cusolverDnHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnDestroy(handle as _))
}

unsafe fn dn_set_stream(handle: cusolverDnHandle_t, streamId: cudaStream_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnSetStream(handle as _, streamId as _))
}

unsafe fn dn_get_stream(
    handle: cusolverDnHandle_t,
    streamId: *mut cudaStream_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnGetStream(handle as _, streamId as _))
}

unsafe fn dn_zzgesv(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuDoubleComplex,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut cuDoubleComplex,
    lddb: cusolver_int_t,
    dX: *mut cuDoubleComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZZgesv(
        handle as _,
        n,
        nrhs,
        dA as _,
        ldda,
        dipiv,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ccgesv(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuComplex,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut cuComplex,
    lddb: cusolver_int_t,
    dX: *mut cuComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCCgesv(
        handle as _,
        n,
        nrhs,
        dA as _,
        ldda as _,
        dipiv,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ddgesv(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f64,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut f64,
    lddb: cusolver_int_t,
    dX: *mut f64,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDDgesv(
        handle as _,
        n,
        nrhs,
        dA,
        ldda,
        dipiv,
        dB as _,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ssgesv(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f32,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut f32,
    lddb: cusolver_int_t,
    dX: *mut f32,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSSgesv(
        handle as _,
        n,
        nrhs,
        dA,
        ldda,
        dipiv,
        dB as _,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_zzgesv_bufferSize(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuDoubleComplex,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut cuDoubleComplex,
    lddb: cusolver_int_t,
    dX: *mut cuDoubleComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZZgesv_bufferSize(
        handle as _,
        n,
        nrhs,
        dA as _,
        ldda,
        dipiv,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_ccgesv_bufferSize(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuComplex,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut cuComplex,
    lddb: cusolver_int_t,
    dX: *mut cuComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCCgesv_bufferSize(
        handle as _,
        n,
        nrhs,
        dA as _,
        ldda,
        dipiv,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_ddgesv_bufferSize(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f64,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut f64,
    lddb: cusolver_int_t,
    dX: *mut f64,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDDgesv_bufferSize(
        handle as _,
        n,
        nrhs,
        dA,
        ldda,
        dipiv,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

pub unsafe fn dn_ssgesv_bufferSize(
    handle: cusolverDnHandle_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f32,
    ldda: cusolver_int_t,
    dipiv: *mut cusolver_int_t,
    dB: *mut f32,
    lddb: cusolver_int_t,
    dX: *mut f32,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSSgesv_bufferSize(
        handle as _,
        n,
        nrhs,
        dA,
        ldda,
        dipiv,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_zzgels(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuDoubleComplex,
    ldda: cusolver_int_t,
    dB: *mut cuDoubleComplex,
    lddb: cusolver_int_t,
    dX: *mut cuDoubleComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZZgels(
        handle as _,
        m,
        n,
        nrhs,
        dA as _,
        ldda,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ccgels(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuComplex,
    ldda: cusolver_int_t,
    dB: *mut cuComplex,
    lddb: cusolver_int_t,
    dX: *mut cuComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCCgels(
        handle as _,
        m,
        n,
        nrhs,
        dA as _,
        ldda,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ddgels(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f64,
    ldda: cusolver_int_t,
    dB: *mut f64,
    lddb: cusolver_int_t,
    dX: *mut f64,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDDgels(
        handle as _,
        m,
        n,
        nrhs,
        dA,
        ldda,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_ssgels(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f32,
    ldda: cusolver_int_t,
    dB: *mut f32,
    lddb: cusolver_int_t,
    dX: *mut f32,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: usize,
    iter: *mut cusolver_int_t,
    d_info: *mut cusolver_int_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSSgels(
        handle as _,
        m,
        n,
        nrhs,
        dA,
        ldda,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
        iter,
        d_info,
    ))
}

unsafe fn dn_zzgels_bufferSize(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuDoubleComplex,
    ldda: cusolver_int_t,
    dB: *mut cuDoubleComplex,
    lddb: cusolver_int_t,
    dX: *mut cuDoubleComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZZgels_bufferSize(
        handle as _,
        m,
        n,
        nrhs,
        dA as _,
        ldda,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_ccgels_bufferSize(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut cuComplex,
    ldda: cusolver_int_t,
    dB: *mut cuComplex,
    lddb: cusolver_int_t,
    dX: *mut cuComplex,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCCgels_bufferSize(
        handle as _,
        m,
        n,
        nrhs,
        dA as _,
        ldda,
        dB as _,
        lddb,
        dX as _,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_ddgels_bufferSize(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f64,
    ldda: cusolver_int_t,
    dB: *mut f64,
    lddb: cusolver_int_t,
    dX: *mut f64,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDDgels_bufferSize(
        handle as _,
        m,
        n,
        nrhs,
        dA,
        ldda,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

unsafe fn dn_ssgels_bufferSize(
    handle: cusolverDnHandle_t,
    m: cusolver_int_t,
    n: cusolver_int_t,
    nrhs: cusolver_int_t,
    dA: *mut f32,
    ldda: cusolver_int_t,
    dB: *mut f32,
    lddb: cusolver_int_t,
    dX: *mut f32,
    lddx: cusolver_int_t,
    dWorkspace: *mut ::std::os::raw::c_void,
    lwork_bytes: *mut usize,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSSgels_bufferSize(
        handle as _,
        m,
        n,
        nrhs,
        dA,
        ldda,
        dB,
        lddb,
        dX,
        lddx,
        dWorkspace,
        lwork_bytes,
    ))
}

pub unsafe fn dn_spotrf_bufferSize(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotrf_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        Lwork,
    ))
}

unsafe fn dn_dpotrf_bufferSize(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotrf_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        Lwork,
    ))
}

unsafe fn dn_cpotrf_bufferSize(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrf_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        Lwork,
    ))
}

unsafe fn dn_zpotrf_bufferSize(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrf_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        Lwork,
    ))
}

pub unsafe fn dn_spotrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    Workspace: *mut f32,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotrf(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        Workspace,
        Lwork,
        devInfo,
    ))
}

pub unsafe fn dn_dpotrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    Workspace: *mut f64,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotrf(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        Workspace,
        Lwork,
        devInfo,
    ))
}

pub unsafe fn dn_cpotrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    Workspace: *mut cuComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrf(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        Workspace as _,
        Lwork,
        devInfo,
    ))
}

pub unsafe fn dn_zpotrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Workspace: *mut cuDoubleComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZpotrf(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        Workspace as _,
        Lwork,
        devInfo,
    ))
}

pub unsafe fn dn_spotrs(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotrs(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        devInfo,
    ))
}

pub unsafe fn dn_dpotrs(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *mut f64,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotrs(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        devInfo,
    ))
}

pub unsafe fn dn_cpotrs(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrs(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A as _,
        lda,
        B as _,
        ldb,
        devInfo,
    ))
}

pub unsafe fn dn_zpotrs(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZpotrs(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A as _,
        lda,
        B as _,
        ldb,
        devInfo,
    ))
}

unsafe fn dn_spotrf_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    Aarray: *mut *mut f32,
    lda: ::std::os::raw::c_int,
    infoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotrfBatched(
        handle as _,
        fill_mode,
        n,
        Aarray,
        lda,
        infoArray,
        batchSize,
    ))
}

unsafe fn dn_dpotrf_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    Aarray: *mut *mut f64,
    lda: ::std::os::raw::c_int,
    infoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotrfBatched(
        handle as _,
        fill_mode,
        n,
        Aarray,
        lda,
        infoArray,
        batchSize,
    ))
}

unsafe fn dn_cpotrf_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    Aarray: *mut *mut cuComplex,
    lda: ::std::os::raw::c_int,
    infoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrfBatched(
        handle as _,
        fill_mode,
        n,
        Aarray as _,
        lda,
        infoArray,
        batchSize,
    ))
}

unsafe fn dn_zpotrf_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    Aarray: *mut *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    infoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZpotrfBatched(
        handle as _,
        fill_mode,
        n,
        Aarray as _,
        lda,
        infoArray,
        batchSize,
    ))
}

unsafe fn dn_spotrs_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *mut *mut f32,
    lda: ::std::os::raw::c_int,
    B: *mut *mut f32,
    ldb: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotrsBatched(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        d_info,
        batchSize,
    ))
}

unsafe fn dn_dpotrs_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *mut *mut f64,
    lda: ::std::os::raw::c_int,
    B: *mut *mut f64,
    ldb: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotrsBatched(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A,
        lda,
        B,
        ldb,
        d_info,
        batchSize,
    ))
}

unsafe fn dn_cpotrs_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *mut *mut cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotrsBatched(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A as _,
        lda,
        B as _,
        ldb,
        d_info,
        batchSize,
    ))
}

unsafe fn dn_zpotrs_batched(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *mut *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZpotrsBatched(
        handle as _,
        fill_mode,
        n,
        nrhs,
        A as _,
        lda,
        B as _,
        ldb,
        d_info,
        batchSize,
    ))
}

unsafe fn dn_spotri_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotri_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        lwork,
    ))
}

unsafe fn dn_dpotri_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotri_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        lwork,
    ))
}

unsafe fn dn_cpotri_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotri_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_zpotri_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZpotri_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_spotri(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSpotri(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        work,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_dpotri(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDpotri(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        work,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_cpotri(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotri(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        work as _,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_zpotri(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCpotri(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        work as _,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_sgetrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgetrf_bufferSize(
        handle as _,
        m,
        n,
        A,
        lda,
        Lwork,
    ))
}

unsafe fn dn_dgetrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgetrf_bufferSize(
        handle as _,
        m,
        n,
        A,
        lda,
        Lwork,
    ))
}

unsafe fn dn_cgetrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgetrf_bufferSize(
        handle as _,
        m,
        n,
        A as _,
        lda,
        Lwork,
    ))
}

unsafe fn dn_zgetrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgetrf_bufferSize(
        handle as _,
        m,
        n,
        A as _,
        lda,
        Lwork,
    ))
}

unsafe fn dn_sgetrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    Workspace: *mut f32,
    devIpiv: *mut ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgetrf(
        handle as _,
        m,
        n,
        A,
        lda,
        Workspace,
        devIpiv,
        devInfo,
    ))
}

unsafe fn dn_dgetrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    Workspace: *mut f64,
    devIpiv: *mut ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgetrf(
        handle as _,
        m,
        n,
        A,
        lda,
        Workspace,
        devIpiv,
        devInfo,
    ))
}

unsafe fn dn_cgetrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    Workspace: *mut cuComplex,
    devIpiv: *mut ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgetrf(
        handle as _,
        m,
        n,
        A as _,
        lda,
        Workspace as _,
        devIpiv,
        devInfo,
    ))
}

unsafe fn dn_zgetrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Workspace: *mut cuDoubleComplex,
    devIpiv: *mut ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgetrf(
        handle as _,
        m,
        n,
        A as _,
        lda,
        Workspace as _,
        devIpiv,
        devInfo,
    ))
}

unsafe fn dn_sgetrs(
    handle: cusolverDnHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnSgetrs(
        handle as _,
        op,
        n,
        nrhs,
        A,
        lda,
        devIpiv,
        B,
        ldb,
        devInfo,
    ))
}

unsafe fn dn_dgetrs(
    handle: cusolverDnHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    B: *mut f64,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnDgetrs(
        handle as _,
        op,
        n,
        nrhs,
        A,
        lda,
        devIpiv,
        B,
        ldb,
        devInfo,
    ))
}

unsafe fn dn_cgetrs(
    handle: cusolverDnHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnCgetrs(
        handle as _,
        op,
        n,
        nrhs,
        A as _,
        lda,
        devIpiv,
        B as _,
        ldb,
        devInfo,
    ))
}

unsafe fn dn_zgetrs(
    handle: cusolverDnHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnZgetrs(
        handle as _,
        op,
        n,
        nrhs,
        A as _,
        lda,
        devIpiv,
        B as _,
        ldb,
        devInfo,
    ))
}

unsafe fn dn_sgeqrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgeqrf_bufferSize(
        handle as _,
        m,
        n,
        A,
        lda,
        lwork,
    ))
}

unsafe fn dn_dgeqrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgeqrf_bufferSize(
        handle as _,
        m,
        n,
        A,
        lda,
        lwork,
    ))
}

unsafe fn dn_cgeqrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgeqrf_bufferSize(
        handle as _,
        m,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_zgeqrf_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgeqrf_bufferSize(
        handle as _,
        m,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_sgeqrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    TAU: *mut f32,
    Workspace: *mut f32,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgeqrf(
        handle as _,
        m,
        n,
        A,
        lda,
        TAU,
        Workspace,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_dgeqrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    TAU: *mut f64,
    Workspace: *mut f64,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgeqrf(
        handle as _,
        m,
        n,
        A,
        lda,
        TAU,
        Workspace,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_cgeqrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    TAU: *mut cuComplex,
    Workspace: *mut cuComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgeqrf(
        handle as _,
        m,
        n,
        A as _,
        lda,
        TAU as _,
        Workspace as _,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_zgeqrf(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    TAU: *mut cuDoubleComplex,
    Workspace: *mut cuDoubleComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgeqrf(
        handle as _,
        m,
        n,
        A as _,
        lda,
        TAU as _,
        Workspace as _,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_sorgqr_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSorgqr_bufferSize(
        handle as _,
        m,
        n,
        k,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_dorgqr_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDorgqr_bufferSize(
        handle as _,
        m,
        n,
        k,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_cungqr_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCungqr_bufferSize(
        handle as _,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_zungqr_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZungqr_bufferSize(
        handle as _,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_sorgqr(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSorgqr(
        handle as _,
        m,
        n,
        k,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dorgqr(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDorgqr(
        handle as _,
        m,
        n,
        k,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cungqr(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCungqr(
        handle as _,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zungqr(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZungqr(
        handle as _,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_sormqr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    C: *const f32,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverSormqr_bufferSize(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_dormqr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    C: *const f64,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDormqr_bufferSize(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_cunmqr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    C: *const cuComplex,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnCunmqr_bufferSize(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_zunmqr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    C: *const cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnZunmqr_bufferSize(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_sormqr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnSormqr(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A,
        lda,
        tau,
        C,
        ldc,
        work,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_dormqr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnDormqr(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A,
        lda,
        tau,
        C,
        ldc,
        work,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_cunmqr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnCunmqr(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        work as _,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_zunmqr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnZunmqr(
        handle as _,
        side_mode,
        op,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        work as _,
        lwork,
        devInfo,
    ))
}

unsafe fn dn_ssytrf_buffer_size(
    handle: cusolverDnHandle_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSsytrf_bufferSize(handle as _, n, A, lda, lwork))
}

unsafe fn dn_dsytrf_buffer_size(
    handle: cusolverDnHandle_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDsytrf_bufferSize(handle as _, n, A, lda, lwork))
}

unsafe fn dn_csytrf_buffer_size(
    handle: cusolverDnHandle_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCsytrf_bufferSize(
        handle as _,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_zsytrf_buffer_size(
    handle: cusolverDnHandle_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZsytrf_bufferSize(
        handle as _,
        n,
        A as _,
        lda,
        lwork,
    ))
}

unsafe fn dn_ssytrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    ipiv: *mut ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsytrf(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        ipiv,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsytrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    ipiv: *mut ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsytrf(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        ipiv,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_csytrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    ipiv: *mut ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCsytrf(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        ipiv,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zsytrf(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    ipiv: *mut ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZsytrf(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        ipiv,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_sgebrd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgebrd_bufferSize(handle as _, m, n, Lwork))
}

unsafe fn dn_dgebrd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgebrd_bufferSize(handle as _, m, n, Lwork))
}

unsafe fn dn_cgebrd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgebrd_bufferSize(handle as _, m, n, Lwork))
}

unsafe fn dn_zgebrd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgebrd_bufferSize(handle as _, m, n, Lwork))
}

unsafe fn dn_sgebrd(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    D: *mut f32,
    E: *mut f32,
    TAUQ: *mut f32,
    TAUP: *mut f32,
    Work: *mut f32,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgebrd(
        handle as _,
        m,
        n,
        A,
        lda,
        D,
        E,
        TAUQ,
        TAUP,
        Work,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_dgebrd(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    D: *mut f64,
    E: *mut f64,
    TAUQ: *mut f64,
    TAUP: *mut f64,
    Work: *mut f64,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgebrd(
        handle as _,
        m,
        n,
        A,
        lda,
        D,
        E,
        TAUQ,
        TAUP,
        Work,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_cgebrd(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    D: *mut f32,
    E: *mut f32,
    TAUQ: *mut cuComplex,
    TAUP: *mut cuComplex,
    Work: *mut cuComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgebrd(
        handle as _,
        m,
        n,
        A as _,
        lda,
        D,
        E,
        TAUQ as _,
        TAUP as _,
        Work as _,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_zgebrd(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    D: *mut f64,
    E: *mut f64,
    TAUQ: *mut cuDoubleComplex,
    TAUP: *mut cuDoubleComplex,
    Work: *mut cuDoubleComplex,
    Lwork: ::std::os::raw::c_int,
    devInfo: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgebrd(
        handle as _,
        m,
        n,
        A as _,
        lda,
        D,
        E,
        TAUQ as _,
        TAUP as _,
        Work as _,
        Lwork,
        devInfo,
    ))
}

unsafe fn dn_sorgbr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnSorgbr_bufferSize(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_dorgbr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnDorgbr_bufferSize(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_cungbr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnCungbr_bufferSize(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_zungbr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnZungbr_bufferSize(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_sorgbr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnSorgbr(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dorgbr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnDorgbr(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cungbr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnCungbr(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zungbr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    to_cuda(hipsolverDnZungbr(
        handle as _,
        side_mode,
        m,
        n,
        k,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_ssytrd_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    d: *const f32,
    e: *const f32,
    tau: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsytrd_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        d,
        e,
        tau,
        lwork,
    ))
}

unsafe fn dn_dsytrd_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    d: *const f64,
    e: *const f64,
    tau: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsytrd_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        d,
        e,
        tau,
        lwork,
    ))
}

unsafe fn dn_chetrd_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    d: *const f32,
    e: *const f32,
    tau: *const cuComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnChetrd_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        d,
        e,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_zhetrd_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    d: *const f64,
    e: *const f64,
    tau: *const cuDoubleComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZhetrd_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        d,
        e,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_ssytrd(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    d: *mut f32,
    e: *mut f32,
    tau: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsytrd(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        d,
        e,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsytrd(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    d: *mut f64,
    e: *mut f64,
    tau: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsytrd(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        d,
        e,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_chetrd(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    d: *mut f32,
    e: *mut f32,
    tau: *mut cuComplex,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnChetrd(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        d,
        e,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zhetrd(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    d: *mut f64,
    e: *mut f64,
    tau: *mut cuDoubleComplex,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZhetrd(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        d,
        e,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_sorgtr_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSorgtr_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_dorgtr_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDorgtr_bufferSize(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        tau,
        lwork,
    ))
}

unsafe fn dn_cungtr_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCungtr_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_zungtr_buffer_size(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZungtr_bufferSize(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        tau as _,
        lwork,
    ))
}

unsafe fn dn_sorgtr(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSorgtr(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dorgtr(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDorgtr(
        handle as _,
        fill_mode,
        n,
        A,
        lda,
        tau,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cungtr(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCungtr(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zungtr(
    handle: cusolverDnHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZungtr(
        handle as _,
        fill_mode,
        n,
        A as _,
        lda,
        tau as _,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_sormtr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    tau: *const f32,
    C: *const f32,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);

    to_cuda(hipsolverDnSormtr_bufferSize(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A,
        lda,
        tau,
        C,
        ldc,
        lwork,
    ))
}

unsafe fn dn_dormtr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    tau: *const f64,
    C: *const f64,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);

    to_cuda(hipsolverDnDormtr_bufferSize(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A,
        lda,
        tau,
        C,
        ldc,
        lwork,
    ))
}

unsafe fn dn_cunmtr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuComplex,
    C: *const cuComplex,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnCunmtr_bufferSize(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_zunmtr_buffer_size(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *const cuDoubleComplex,
    C: *const cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnZunmtr_bufferSize(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        lwork,
    ))
}

unsafe fn dn_sormtr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    tau: *mut f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnSormtr(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A,
        lda,
        tau,
        C,
        ldc,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dormtr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    tau: *mut f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnDormtr(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A,
        lda,
        tau,
        C,
        ldc,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cunmtr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    tau: *mut cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnCunmtr(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zunmtr(
    handle: cusolverDnHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    tau: *mut cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let side_mode = to_side_mode(side);
    let fill_mode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(hipsolverDnZunmtr(
        handle as _,
        side_mode,
        fill_mode,
        op,
        m,
        n,
        A as _,
        lda,
        tau as _,
        C as _,
        ldc,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_sgesvd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgesvd_bufferSize(handle as _, m, n, lwork))
}

unsafe fn dn_dgesvd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgesvd_bufferSize(handle as _, m, n, lwork))
}

unsafe fn dn_cgesvd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgesvd_bufferSize(handle as _, m, n, lwork))
}

unsafe fn dn_zgesvd_buffer_size(
    handle: cusolverDnHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgesvd_bufferSize(handle as _, m, n, lwork))
}

unsafe fn dn_sgesvd(
    handle: cusolverDnHandle_t,
    jobu: ::std::os::raw::c_schar,
    jobvt: ::std::os::raw::c_schar,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut f32,
    ldu: ::std::os::raw::c_int,
    VT: *mut f32,
    ldvt: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    rwork: *mut f32,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnSgesvd(
        handle as _,
        jobu,
        jobvt,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        VT,
        ldvt,
        work,
        lwork,
        rwork,
        info,
    ))
}

unsafe fn dn_dgesvd(
    handle: cusolverDnHandle_t,
    jobu: ::std::os::raw::c_schar,
    jobvt: ::std::os::raw::c_schar,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut f64,
    ldu: ::std::os::raw::c_int,
    VT: *mut f64,
    ldvt: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    rwork: *mut f64,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnDgesvd(
        handle as _,
        jobu,
        jobvt,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        VT,
        ldvt,
        work,
        lwork,
        rwork,
        info,
    ))
}

unsafe fn dn_cgesvd(
    handle: cusolverDnHandle_t,
    jobu: ::std::os::raw::c_schar,
    jobvt: ::std::os::raw::c_schar,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut cuComplex,
    ldu: ::std::os::raw::c_int,
    VT: *mut cuComplex,
    ldvt: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    rwork: *mut f32,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnCgesvd(
        handle as _,
        jobu,
        jobvt,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        VT as _,
        ldvt,
        work as _,
        lwork,
        rwork,
        info,
    ))
}

unsafe fn dn_zgesvd(
    handle: cusolverDnHandle_t,
    jobu: ::std::os::raw::c_schar,
    jobvt: ::std::os::raw::c_schar,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    VT: *mut cuDoubleComplex,
    ldvt: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    rwork: *mut f64,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnZgesvd(
        handle as _,
        jobu,
        jobvt,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        VT as _,
        ldvt,
        work as _,
        lwork,
        rwork,
        info,
    ))
}

unsafe fn dn_ssyevd_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevd_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
    ))
}

unsafe fn dn_dsyevd_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevd_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
    ))
}

unsafe fn dn_cheevd_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevd_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        lwork,
    ))
}

unsafe fn dn_zheevd_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevd_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        lwork,
    ))
}

unsafe fn dn_ssyevd(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevd(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsyevd(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevd(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cheevd(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevd(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zheevd(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevd(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_ssyevdx_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnSsyevdx_bufferSize(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_dsyevdx_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnDsyevdx_bufferSize(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_cheevdx_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnCheevdx_bufferSize(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_zheevdx_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnZheevdx_bufferSize(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_ssyevdx(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnSsyevdx(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsyevdx(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnDsyevdx(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_cheevdx(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnCheevdx(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zheevdx(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    to_cuda(hipsolverDnZheevdx(
        handle as _,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_ssygvdx_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvdx_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_dsygvdx_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvdx_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_chegvdx_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvdx_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_zhegvdx_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvdx_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        lwork,
    ))
}

unsafe fn dn_ssygvdx(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvdx(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsygvdx(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    B: *mut f64,
    ldb: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvdx(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_chegvdx(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    vl: f32,
    vu: f32,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvdx(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zhegvdx(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    range: cusolverEigRange_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    vl: f64,
    vu: f64,
    il: ::std::os::raw::c_int,
    iu: ::std::os::raw::c_int,
    meig: *mut ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_range = to_eig_range(range);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvdx(
        handle as _,
        eig_type,
        eig_mode,
        eig_range,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        vl,
        vu,
        il,
        iu,
        meig,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_ssygvd_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvd_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        lwork,
    ))
}

unsafe fn dn_dsygvd_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvd_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        lwork,
    ))
}

unsafe fn dn_chegvd_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvd_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        lwork,
    ))
}

unsafe fn dn_zhegvd_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvd_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        lwork,
    ))
}

unsafe fn dn_ssygvd(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvd(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_dsygvd(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    B: *mut f64,
    ldb: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvd(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        work,
        lwork,
        info,
    ))
}

unsafe fn dn_chegvd(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvd(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_zhegvd(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvd(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        work as _,
        lwork,
        info,
    ))
}

unsafe fn dn_create_syevj_info(info: *mut syevjInfo_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnCreateSyevjInfo(info as _))
}

unsafe fn dn_destroy_syevj_info(info: syevjInfo_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnDestroySyevjInfo(info as _))
}

unsafe fn dn_xsyevj_set_tolerance(info: syevjInfo_t, tolerance: f64) -> cusolverStatus_t {
    to_cuda(hipsolverDnXsyevjSetTolerance(info as _, tolerance))
}

unsafe fn dn_xsyevj_set_max_sweeps(
    info: syevjInfo_t,
    max_sweeps: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXsyevjSetMaxSweeps(info as _, max_sweeps))
}

unsafe fn dn_xsyevj_set_sort_eig(
    info: syevjInfo_t,
    sort_eig: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXsyevjSetSortEig(info as _, sort_eig))
}

unsafe fn dn_xsyevj_get_residual(
    handle: cusolverDnHandle_t,
    info: syevjInfo_t,
    residual: *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXsyevjGetResidual(
        handle as _,
        info as _,
        residual,
    ))
}

unsafe fn dn_xsyevj_get_sweeps(
    handle: cusolverDnHandle_t,
    info: syevjInfo_t,
    executed_sweeps: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXsyevjGetSweeps(
        handle as _,
        info as _,
        executed_sweeps,
    ))
}

unsafe fn dn_ssyevj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevjBatched_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_dsyevj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevjBatched_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_cheevj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevjBatched_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W as _,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_zheevj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevjBatched_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W as _,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_ssyevj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevjBatched(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_dsyevj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevjBatched(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_cheevj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevjBatched(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_zheevj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevjBatched(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_ssyevj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevj_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_dsyevj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevj_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_cheevj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevj_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_zheevj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevj_bufferSize(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_ssyevj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnSsyevj(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_dsyevj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnDsyevj(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        W,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_cheevj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnCheevj(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_zheevj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    to_cuda(hipsolverDnZheevj(
        handle as _,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        W,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_ssygvj_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvj_bufferSize(
        handle as _,
        eig_type as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_dsygvj_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvj_bufferSize(
        handle as _,
        eig_type as _,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_chegvj_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    W: *const f32,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvj_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_zhegvj_buffer_size(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    W: *const f64,
    lwork: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvj_bufferSize(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        lwork,
        params as _,
    ))
}

unsafe fn dn_ssygvj(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnSsygvj(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_dsygvj(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    B: *mut f64,
    ldb: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnDsygvj(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A,
        lda,
        B,
        ldb,
        W,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_chegvj(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    W: *mut f32,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnChegvj(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_zhegvj(
    handle: cusolverDnHandle_t,
    itype: cusolverEigType_t,
    jobz: cusolverEigMode_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    W: *mut f64,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: syevjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    let fill_mode = to_fill(uplo);
    let eig_type = to_eig_type(itype);
    to_cuda(hipsolverDnZhegvj(
        handle as _,
        eig_type,
        eig_mode,
        fill_mode,
        n,
        A as _,
        lda,
        B as _,
        ldb,
        W,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_create_gesvdj_info(info: *mut gesvdjInfo_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnCreateGesvdjInfo(info as _))
}

unsafe fn dn_destroy_gesvdj_info(info: gesvdjInfo_t) -> cusolverStatus_t {
    to_cuda(hipsolverDnDestroyGesvdjInfo(info as _))
}

unsafe fn dn_xgesvdj_set_tolerance(info: gesvdjInfo_t, tolerance: f64) -> cusolverStatus_t {
    to_cuda(hipsolverDnXgesvdjSetTolerance(info as _, tolerance))
}

unsafe fn dn_xgesvdj_set_max_sweeps(
    info: gesvdjInfo_t,
    max_sweeps: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXgesvdjSetMaxSweeps(info as _, max_sweeps))
}

unsafe fn dn_xgesvdj_set_sort_eig(
    info: gesvdjInfo_t,
    sort_svd: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXgesvdjSetSortEig(info as _, sort_svd))
}

unsafe fn dn_xgesvdj_get_residual(
    handle: cusolverDnHandle_t,
    info: gesvdjInfo_t,
    residual: *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXgesvdjGetResidual(
        handle as _,
        info as _,
        residual,
    ))
}

unsafe fn dn_xgesvdj_get_sweeps(
    handle: cusolverDnHandle_t,
    info: gesvdjInfo_t,
    executed_sweeps: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverDnXgesvdjGetSweeps(
        handle as _,
        info as _,
        executed_sweeps,
    ))
}

unsafe fn dn_sgesvdj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    S: *const f32,
    U: *const f32,
    ldu: ::std::os::raw::c_int,
    V: *const f32,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdjBatched_bufferSize(
        handle as _,
        eig_mode,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_dgesvdj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    S: *const f64,
    U: *const f64,
    ldu: ::std::os::raw::c_int,
    V: *const f64,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdjBatched_bufferSize(
        handle as _,
        eig_mode,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_cgesvdj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    S: *const f32,
    U: *const cuComplex,
    ldu: ::std::os::raw::c_int,
    V: *const cuComplex,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdjBatched_bufferSize(
        handle as _,
        eig_mode,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_zgesvdj_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    S: *const f64,
    U: *const cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    V: *const cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdjBatched_bufferSize(
        handle as _,
        eig_mode,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        lwork,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_sgesvdj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut f32,
    ldu: ::std::os::raw::c_int,
    V: *mut f32,
    ldv: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdjBatched(
        handle as _,
        eig_mode,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        work,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_dgesvdj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut f64,
    ldu: ::std::os::raw::c_int,
    V: *mut f64,
    ldv: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdjBatched(
        handle as _,
        eig_mode,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        work,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_cgesvdj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut cuComplex,
    ldu: ::std::os::raw::c_int,
    V: *mut cuComplex,
    ldv: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdjBatched(
        handle as _,
        eig_mode,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        work as _,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_zgesvdj_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    V: *mut cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdjBatched(
        handle as _,
        eig_mode,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        work as _,
        lwork,
        info,
        params as _,
        batchSize,
    ))
}

unsafe fn dn_sgesvdj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    S: *const f32,
    U: *const f32,
    ldu: ::std::os::raw::c_int,
    V: *const f32,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdj_bufferSize(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        lwork,
        params as _,
    ))
}

unsafe fn dn_dgesvdj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    S: *const f64,
    U: *const f64,
    ldu: ::std::os::raw::c_int,
    V: *const f64,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdj_bufferSize(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        lwork,
        params as _,
    ))
}

unsafe fn dn_cgesvdj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    S: *const f32,
    U: *const cuComplex,
    ldu: ::std::os::raw::c_int,
    V: *const cuComplex,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdj_bufferSize(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        lwork,
        params as _,
    ))
}

unsafe fn dn_zgesvdj_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    S: *const f64,
    U: *const cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    V: *const cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    lwork: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdj_bufferSize(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        lwork,
        params as _,
    ))
}

unsafe fn dn_sgesvdj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut f32,
    ldu: ::std::os::raw::c_int,
    V: *mut f32,
    ldv: ::std::os::raw::c_int,
    work: *mut f32,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdj(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_dgesvdj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut f64,
    ldu: ::std::os::raw::c_int,
    V: *mut f64,
    ldv: ::std::os::raw::c_int,
    work: *mut f64,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdj(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        work,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_cgesvdj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f32,
    U: *mut cuComplex,
    ldu: ::std::os::raw::c_int,
    V: *mut cuComplex,
    ldv: ::std::os::raw::c_int,
    work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdj(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_zgesvdj(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    econ: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    S: *mut f64,
    U: *mut cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    V: *mut cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    params: gesvdjInfo_t,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdj(
        handle as _,
        eig_mode,
        econ,
        m,
        n,
        A as _,
        lda,
        S,
        U as _,
        ldu,
        V as _,
        ldv,
        work as _,
        lwork,
        info,
        params as _,
    ))
}

unsafe fn dn_sgesvda_strided_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const f32,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *const f32,
    strideS: ::std::os::raw::c_longlong,
    d_U: *const f32,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *const f32,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    lwork: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdaStridedBatched_bufferSize(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A,
        lda,
        strideA,
        d_S,
        strideS,
        d_U,
        ldu,
        strideU,
        d_V,
        ldv,
        strideV,
        lwork,
        batchSize,
    ))
}

unsafe fn dn_dgesvda_strided_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const f64,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *const f64,
    strideS: ::std::os::raw::c_longlong,
    d_U: *const f64,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *const f64,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    lwork: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdaStridedBatched_bufferSize(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A,
        lda,
        strideA,
        d_S,
        strideS,
        d_U,
        ldu,
        strideU,
        d_V,
        ldv,
        strideV,
        lwork,
        batchSize,
    ))
}

unsafe fn dn_cgesvda_strided_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *const f32,
    strideS: ::std::os::raw::c_longlong,
    d_U: *const cuComplex,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *const cuComplex,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    lwork: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdaStridedBatched_bufferSize(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A as _,
        lda,
        strideA,
        d_S as _,
        strideS,
        d_U as _,
        ldu,
        strideU,
        d_V as _,
        ldv,
        strideV,
        lwork,
        batchSize,
    ))
}

unsafe fn dn_zgesvda_strided_batched_buffer_size(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *const f64,
    strideS: ::std::os::raw::c_longlong,
    d_U: *const cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *const cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    lwork: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdaStridedBatched_bufferSize(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A as _,
        lda,
        strideA,
        d_S as _,
        strideS,
        d_U as _,
        ldu,
        strideU,
        d_V as _,
        ldv,
        strideV,
        lwork,
        batchSize,
    ))
}

unsafe fn dn_sgesvda_strided_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const f32,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *mut f32,
    strideS: ::std::os::raw::c_longlong,
    d_U: *mut f32,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *mut f32,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    d_work: *mut f32,
    lwork: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    h_R_nrmF: *mut f64,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnSgesvdaStridedBatched(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A,
        lda,
        strideA,
        d_S,
        strideS,
        d_U,
        ldu,
        strideU,
        d_V,
        ldv,
        strideV,
        d_work,
        lwork,
        d_info,
        h_R_nrmF,
        batchSize,
    ))
}

unsafe fn dn_dgesvda_strided_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const f64,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *mut f64,
    strideS: ::std::os::raw::c_longlong,
    d_U: *mut f64,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *mut f64,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    d_work: *mut f64,
    lwork: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    h_R_nrmF: *mut f64,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnDgesvdaStridedBatched(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A,
        lda,
        strideA,
        d_S,
        strideS,
        d_U,
        ldu,
        strideU,
        d_V,
        ldv,
        strideV,
        d_work,
        lwork,
        d_info,
        h_R_nrmF,
        batchSize,
    ))
}

unsafe fn dn_cgesvda_strided_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *mut f32,
    strideS: ::std::os::raw::c_longlong,
    d_U: *mut cuComplex,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *mut cuComplex,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    d_work: *mut cuComplex,
    lwork: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    h_R_nrmF: *mut f64,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnCgesvdaStridedBatched(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A as _,
        lda,
        strideA,
        d_S as _,
        strideS,
        d_U as _,
        ldu,
        strideU,
        d_V as _,
        ldv,
        strideV,
        d_work as _,
        lwork,
        d_info,
        h_R_nrmF,
        batchSize,
    ))
}

unsafe fn dn_zgesvda_strided_batched(
    handle: cusolverDnHandle_t,
    jobz: cusolverEigMode_t,
    rank: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    d_A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    d_S: *mut f64,
    strideS: ::std::os::raw::c_longlong,
    d_U: *mut cuDoubleComplex,
    ldu: ::std::os::raw::c_int,
    strideU: ::std::os::raw::c_longlong,
    d_V: *mut cuDoubleComplex,
    ldv: ::std::os::raw::c_int,
    strideV: ::std::os::raw::c_longlong,
    d_work: *mut cuDoubleComplex,
    lwork: ::std::os::raw::c_int,
    d_info: *mut ::std::os::raw::c_int,
    h_R_nrmF: *mut f64,
    batchSize: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    let eig_mode = to_eig_mode(jobz);
    to_cuda(hipsolverDnZgesvdaStridedBatched(
        handle as _,
        eig_mode,
        rank,
        m,
        n,
        d_A as _,
        lda,
        strideA,
        d_S as _,
        strideS,
        d_U as _,
        ldu,
        strideU,
        d_V as _,
        ldv,
        strideV,
        d_work as _,
        lwork,
        d_info,
        h_R_nrmF,
        batchSize,
    ))
}

unsafe fn rf_create(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfCreate(handle as _))
}

unsafe fn rf_destroy(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfDestroy(handle as _))
}

unsafe fn rf_get_matrix_format(
    handle: cusolverRfHandle_t,
    format: *mut cusolverRfMatrixFormat_t,
    diag: *mut cusolverRfUnitDiagonal_t,
) -> cusolverStatus_t {
    let mut hip_format = hipsolverRfMatrixFormat_t::HIPSOLVERRF_MATRIX_FORMAT_CSC;
    let mut hip_diag = hipsolverRfUnitDiagonal_t::HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
    let status = hipsolverRfGetMatrixFormat(handle as _, &mut hip_format, &mut hip_diag);

    *format = from_hip_to_rf_matrix(hip_format);
    *diag = from_hip_to_rf_dia(hip_diag);
    to_cuda(status)
}

unsafe fn rf_set_matrix_format(
    handle: cusolverRfHandle_t,
    format: cusolverRfMatrixFormat_t,
    diag: cusolverRfUnitDiagonal_t,
) -> cusolverStatus_t {
    let rf_format = to_rf_matrix(format);
    let rf_diag = to_rf_dia(diag);
    to_cuda(hipsolverRfSetMatrixFormat(handle as _, rf_format, rf_diag))
}

unsafe fn rf_set_numeric_properties(
    handle: cusolverRfHandle_t,
    zero: f64,
    boost: f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfSetNumericProperties(handle as _, zero, boost))
}

unsafe fn rf_get_numeric_properties(
    handle: cusolverRfHandle_t,
    zero: *mut f64,
    boost: *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfGetNumericProperties(handle as _, zero, boost))
}

unsafe fn rf_get_numeric_boost_report(
    handle: cusolverRfHandle_t,
    report: *mut cusolverRfNumericBoostReport_t,
) -> cusolverStatus_t {
    let mut hip_report = hipsolverRfNumericBoostReport_t::HIPSOLVERRF_NUMERIC_BOOST_NOT_USED;
    let status = hipsolverRfGetNumericBoostReport(handle as _, &mut hip_report);
    *report = from_hip_to_rf_report(hip_report);
    to_cuda(status)
}

unsafe fn rf_set_algs(
    handle: cusolverRfHandle_t,
    factAlg: cusolverRfFactorization_t,
    solveAlg: cusolverRfTriangularSolve_t,
) -> cusolverStatus_t {
    let hip_fact = to_rf_factor(factAlg);
    let hip_solve = to_rf_solve(solveAlg);
    to_cuda(hipsolverRfSetAlgs(handle as _, hip_fact, hip_solve))
}

unsafe fn rf_get_algs(
    handle: cusolverRfHandle_t,
    factAlg: *mut cusolverRfFactorization_t,
    solveAlg: *mut cusolverRfTriangularSolve_t,
) -> cusolverStatus_t {
    let mut hip_fact = hipsolverRfFactorization_t::HIPSOLVERRF_FACTORIZATION_ALG0;
    let mut hip_solve = hipsolverRfTriangularSolve_t::HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1;
    let status = hipsolverRfGet_Algs(handle as _, &mut hip_fact, &mut hip_solve);
    *factAlg = from_hip_to_rf_factor(hip_fact);
    *solveAlg = from_hip_to_rf_solve(hip_solve);
    to_cuda(status)
}

unsafe fn rf_get_reset_values_fast_mode(
    handle: cusolverRfHandle_t,
    fastMode: *mut cusolverRfResetValuesFastMode_t,
) -> cusolverStatus_t {
    let mut hip_mode = hipsolverRfResetValuesFastMode_t::HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
    let status = hipsolverRfGetResetValuesFastMode(handle as _, &mut hip_mode);
    *fastMode = from_hip_to_rf_mode(hip_mode);
    to_cuda(status)
}

unsafe fn rf_set_reset_values_fast_mode(
    handle: cusolverRfHandle_t,
    fastMode: cusolverRfResetValuesFastMode_t,
) -> cusolverStatus_t {
    let hip_mode = to_rf_mode(fastMode);
    to_cuda(hipsolverRfSetResetValuesFastMode(handle as _, hip_mode))
}

unsafe fn rf_setup_host(
    n: ::std::os::raw::c_int,
    nnzA: ::std::os::raw::c_int,
    h_csrRowPtrA: *mut ::std::os::raw::c_int,
    h_csrColIndA: *mut ::std::os::raw::c_int,
    h_csrValA: *mut f64,
    nnzL: ::std::os::raw::c_int,
    h_csrRowPtrL: *mut ::std::os::raw::c_int,
    h_csrColIndL: *mut ::std::os::raw::c_int,
    h_csrValL: *mut f64,
    nnzU: ::std::os::raw::c_int,
    h_csrRowPtrU: *mut ::std::os::raw::c_int,
    h_csrColIndU: *mut ::std::os::raw::c_int,
    h_csrValU: *mut f64,
    h_P: *mut ::std::os::raw::c_int,
    h_Q: *mut ::std::os::raw::c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfSetupHost(
        n,
        nnzA,
        h_csrRowPtrA,
        h_csrColIndA,
        h_csrValA,
        nnzL,
        h_csrRowPtrL,
        h_csrColIndL,
        h_csrValL,
        nnzU,
        h_csrRowPtrU,
        h_csrColIndU,
        h_csrValU,
        h_P,
        h_Q,
        handle as _,
    ))
}

unsafe fn rf_setup_device(
    n: ::std::os::raw::c_int,
    nnzA: ::std::os::raw::c_int,
    csrRowPtrA: *mut ::std::os::raw::c_int,
    csrColIndA: *mut ::std::os::raw::c_int,
    csrValA: *mut f64,
    nnzL: ::std::os::raw::c_int,
    csrRowPtrL: *mut ::std::os::raw::c_int,
    csrColIndL: *mut ::std::os::raw::c_int,
    csrValL: *mut f64,
    nnzU: ::std::os::raw::c_int,
    csrRowPtrU: *mut ::std::os::raw::c_int,
    csrColIndU: *mut ::std::os::raw::c_int,
    csrValU: *mut f64,
    P: *mut ::std::os::raw::c_int,
    Q: *mut ::std::os::raw::c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfSetupDevice(
        n,
        nnzA,
        csrRowPtrA,
        csrColIndA,
        csrValA,
        nnzL,
        csrRowPtrL,
        csrColIndL,
        csrValL,
        nnzU,
        csrRowPtrU,
        csrColIndU,
        csrValU,
        P,
        Q,
        handle as _,
    ))
}

unsafe fn rf_reset_values(
    n: ::std::os::raw::c_int,
    nnzA: ::std::os::raw::c_int,
    csrRowPtrA: *mut ::std::os::raw::c_int,
    csrColIndA: *mut ::std::os::raw::c_int,
    csrValA: *mut f64,
    P: *mut ::std::os::raw::c_int,
    Q: *mut ::std::os::raw::c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfResetValues(
        n,
        nnzA,
        csrRowPtrA,
        csrColIndA,
        csrValA,
        P,
        Q,
        handle as _,
    ))
}

unsafe fn rf_analyze(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfAnalyze(handle as _))
}

unsafe fn rf_refactor(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfRefactor(handle as _))
}

unsafe fn rf_access_bundled_factors_device(
    handle: cusolverRfHandle_t,
    nnzM: *mut ::std::os::raw::c_int,
    Mp: *mut *mut ::std::os::raw::c_int,
    Mi: *mut *mut ::std::os::raw::c_int,
    Mx: *mut *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfAccessBundledFactorsDevice(
        handle as _,
        nnzM,
        Mp,
        Mi,
        Mx,
    ))
}

unsafe fn rf_extract_bundled_factors_host(
    handle: cusolverRfHandle_t,
    h_nnzM: *mut ::std::os::raw::c_int,
    h_Mp: *mut *mut ::std::os::raw::c_int,
    h_Mi: *mut *mut ::std::os::raw::c_int,
    h_Mx: *mut *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfExtractBundledFactorsHost(
        handle as _,
        h_nnzM,
        h_Mp,
        h_Mi,
        h_Mx,
    ))
}

unsafe fn rf_extract_split_factors_host(
    handle: cusolverRfHandle_t,
    h_nnzL: *mut ::std::os::raw::c_int,
    h_csrRowPtrL: *mut *mut ::std::os::raw::c_int,
    h_csrColIndL: *mut *mut ::std::os::raw::c_int,
    h_csrValL: *mut *mut f64,
    h_nnzU: *mut ::std::os::raw::c_int,
    h_csrRowPtrU: *mut *mut ::std::os::raw::c_int,
    h_csrColIndU: *mut *mut ::std::os::raw::c_int,
    h_csrValU: *mut *mut f64,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfExtractSplitFactorsHost(
        handle as _,
        h_nnzL,
        h_csrRowPtrL,
        h_csrColIndL,
        h_csrValL,
        h_nnzU,
        h_csrRowPtrU,
        h_csrColIndU,
        h_csrValU,
    ))
}

unsafe fn rf_solve(
    handle: cusolverRfHandle_t,
    P: *mut ::std::os::raw::c_int,
    Q: *mut ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Temp: *mut f64,
    ldt: ::std::os::raw::c_int,
    XF: *mut f64,
    ldxf: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfSolve(
        handle as _,
        P,
        Q,
        nrhs,
        Temp,
        ldt,
        XF,
        ldxf,
    ))
}

unsafe fn rf_batch_setup_host(
    batchSize: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nnzA: ::std::os::raw::c_int,
    h_csrRowPtrA: *mut ::std::os::raw::c_int,
    h_csrColIndA: *mut ::std::os::raw::c_int,
    h_csrValA_array: *mut *mut f64,
    nnzL: ::std::os::raw::c_int,
    h_csrRowPtrL: *mut ::std::os::raw::c_int,
    h_csrColIndL: *mut ::std::os::raw::c_int,
    h_csrValL: *mut f64,
    nnzU: ::std::os::raw::c_int,
    h_csrRowPtrU: *mut ::std::os::raw::c_int,
    h_csrColIndU: *mut ::std::os::raw::c_int,
    h_csrValU: *mut f64,
    h_P: *mut ::std::os::raw::c_int,
    h_Q: *mut ::std::os::raw::c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchSetupHost(
        batchSize,
        n,
        nnzA,
        h_csrRowPtrA,
        h_csrColIndA,
        h_csrValA_array,
        nnzL,
        h_csrRowPtrL,
        h_csrColIndL,
        h_csrValL,
        nnzU,
        h_csrRowPtrU,
        h_csrColIndU,
        h_csrValU,
        h_P,
        h_Q,
        handle as _,
    ))
}

unsafe fn rf_batch_reset_values(
    batchSize: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nnzA: ::std::os::raw::c_int,
    csrRowPtrA: *mut ::std::os::raw::c_int,
    csrColIndA: *mut ::std::os::raw::c_int,
    csrValA_array: *mut *mut f64,
    P: *mut ::std::os::raw::c_int,
    Q: *mut ::std::os::raw::c_int,
    handle: cusolverRfHandle_t,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchResetValues(
        batchSize,
        n,
        nnzA,
        csrRowPtrA,
        csrColIndA,
        csrValA_array,
        P,
        Q,
        handle as _,
    ))
}

unsafe fn rf_batch_analyze(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchAnalyze(handle as _))
}

unsafe fn rf_batch_refactor(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchRefactor(handle as _))
}

unsafe fn rf_batch_solve(
    handle: cusolverRfHandle_t,
    P: *mut ::std::os::raw::c_int,
    Q: *mut ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Temp: *mut f64,
    ldt: ::std::os::raw::c_int,
    XF_array: *mut *mut f64,
    ldxf: ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchSolve(
        handle as _,
        P,
        Q,
        nrhs,
        Temp,
        ldt,
        XF_array,
        ldxf,
    ))
}

unsafe fn rf_batch_zero_pivot(
    handle: cusolverRfHandle_t,
    position: *mut ::std::os::raw::c_int,
) -> cusolverStatus_t {
    to_cuda(hipsolverRfBatchZeroPivot(handle as _, position))
}
