#![allow(non_snake_case)]
#[allow(warnings)]
mod cusolver;
pub use cusolver::*;

use cuda_types::*;
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
