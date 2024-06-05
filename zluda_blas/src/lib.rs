#![allow(warnings)]
mod cublas;

pub use cublas::*;

use cuda_types::*;
use rocblas_sys::*;
use rocsolver_sys::{
    rocsolver_cgetrf_batched, rocsolver_cgetri_batched, rocsolver_cgetri_outofplace_batched,
    rocsolver_sgetrf_batched, rocsolver_sgetrf_npvt_batched, rocsolver_zgetrf_batched,
    rocsolver_zgetri_batched, rocsolver_zgetri_outofplace_batched,
};
use std::{mem, ptr};

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> cublasStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

fn to_cuda(status: rocblas_sys::rocblas_status) -> cublasStatus_t {
    match status {
        rocblas_sys::rocblas_status::rocblas_status_success => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocblas_sys::rocblas_status::rocblas_status_size_unchanged => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocblas_sys::rocblas_status::rocblas_status_size_increased => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocblas_sys::rocblas_status::rocblas_status_invalid_handle => {
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED
        }
        rocblas_sys::rocblas_status::rocblas_status_not_implemented => {
            cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
        }
        rocblas_sys::rocblas_status::rocblas_status_invalid_pointer => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocblas_sys::rocblas_status::rocblas_status_invalid_size => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocblas_sys::rocblas_status::rocblas_status_invalid_value => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocblas_sys::rocblas_status::rocblas_status_invalid_value => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocblas_sys::rocblas_status::rocblas_status_memory_error => {
            cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR
        }
        _ => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
    }
}

fn rocsolver_to_cuda(status: rocsolver_sys::rocsolver_status) -> cublasStatus_t {
    match status {
        rocsolver_sys::rocblas_status::rocblas_status_success => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocsolver_sys::rocblas_status::rocblas_status_size_unchanged => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocsolver_sys::rocblas_status::rocblas_status_size_increased => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        rocsolver_sys::rocblas_status::rocblas_status_invalid_handle => {
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED
        }
        rocsolver_sys::rocblas_status::rocblas_status_not_implemented => {
            cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
        }
        rocsolver_sys::rocblas_status::rocblas_status_invalid_pointer => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocsolver_sys::rocblas_status::rocblas_status_invalid_size => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocsolver_sys::rocblas_status::rocblas_status_invalid_value => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocsolver_sys::rocblas_status::rocblas_status_invalid_value => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        rocsolver_sys::rocblas_status::rocblas_status_memory_error => {
            cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR
        }
        _ => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
    }
}

fn to_cuda_solver(status: rocsolver_sys::rocblas_status) -> cublasStatus_t {
    match status {
        rocsolver_sys::rocblas_status::rocblas_status_success => {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
        _ => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
    }
}

unsafe fn create(handle: *mut cublasHandle_t) -> cublasStatus_t {
    to_cuda(rocblas_sys::rocblas_create_handle(handle as _))
}

unsafe fn sgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    x: *const f32,
    incx: i32,
    beta: *const f32,
    y: *mut f32,
    incy: i32,
) -> cublasStatus_t {
    let trans = op_from_cuda(trans);
    to_cuda(rocblas_sgemv(
        handle as _,
        trans,
        m,
        n,
        alpha,
        a,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

fn op_from_cuda(trans: cublasOperation_t) -> rocblas_operation {
    match trans {
        cublasOperation_t::CUBLAS_OP_N => rocblas_operation::rocblas_operation_none,
        cublasOperation_t::CUBLAS_OP_T => rocblas_operation::rocblas_operation_transpose,
        cublasOperation_t::CUBLAS_OP_C => rocblas_operation::rocblas_operation_conjugate_transpose,
        _ => panic!(),
    }
}

fn rocsolver_op_from_cuda(trans: cublasOperation_t) -> rocsolver_sys::rocblas_operation {
    match trans {
        cublasOperation_t::CUBLAS_OP_N => rocsolver_sys::rocblas_operation::rocblas_operation_none,
        cublasOperation_t::CUBLAS_OP_T => {
            rocsolver_sys::rocblas_operation::rocblas_operation_transpose
        }
        cublasOperation_t::CUBLAS_OP_C => {
            rocsolver_sys::rocblas_operation::rocblas_operation_conjugate_transpose
        }
        _ => panic!(),
    }
}

unsafe fn destroy(handle: cublasHandle_t) -> cublasStatus_t {
    to_cuda(rocblas_destroy_handle(handle as _))
}

unsafe fn sgemm_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const std::ffi::c_void,
    atype: cudaDataType,
    lda: i32,
    b: *const std::ffi::c_void,
    btype: cudaDataType,
    ldb: i32,
    beta: *const f32,
    c: *mut std::ffi::c_void,
    ctype: cudaDataType,
    ldc: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    let a_type = type_from_cuda(atype);
    let b_type = type_from_cuda(btype);
    let c_type = type_from_cuda(ctype);
    to_cuda(rocblas_gemm_ex(
        handle as _,
        transa,
        transb,
        m,
        n,
        k,
        alpha as _,
        a as _,
        a_type,
        lda,
        b as _,
        b_type,
        ldb,
        beta as _,
        c as _,
        c_type,
        ldc,
        c as _,
        c_type,
        ldc,
        rocblas_datatype::rocblas_datatype_f32_r,
        rocblas_gemm_algo::rocblas_gemm_algo_standard,
        0,
        0,
    ))
}

fn type_from_cuda(type_: cudaDataType_t) -> rocblas_datatype {
    match type_ {
        cudaDataType_t::CUDA_R_16F => rocblas_datatype::rocblas_datatype_f16_r,
        cudaDataType_t::CUDA_R_32F => rocblas_datatype::rocblas_datatype_f32_r,
        cudaDataType_t::CUDA_R_64F => rocblas_datatype::rocblas_datatype_f64_r,
        cudaDataType_t::CUDA_C_16F => rocblas_datatype::rocblas_datatype_f16_c,
        cudaDataType_t::CUDA_C_32F => rocblas_datatype::rocblas_datatype_f32_c,
        cudaDataType_t::CUDA_C_64F => rocblas_datatype::rocblas_datatype_f64_c,
        cudaDataType_t::CUDA_R_8I => rocblas_datatype::rocblas_datatype_i8_r,
        cudaDataType_t::CUDA_R_8U => rocblas_datatype::rocblas_datatype_u8_r,
        cudaDataType_t::CUDA_R_32I => rocblas_datatype::rocblas_datatype_i32_r,
        cudaDataType_t::CUDA_R_32U => rocblas_datatype::rocblas_datatype_u32_r,
        cudaDataType_t::CUDA_C_8I => rocblas_datatype::rocblas_datatype_i8_c,
        cudaDataType_t::CUDA_C_8U => rocblas_datatype::rocblas_datatype_u8_c,
        cudaDataType_t::CUDA_C_32I => rocblas_datatype::rocblas_datatype_i32_c,
        cudaDataType_t::CUDA_C_32U => rocblas_datatype::rocblas_datatype_u32_c,
        cudaDataType_t::CUDA_R_16BF => rocblas_datatype::rocblas_datatype_bf16_r,
        cudaDataType_t::CUDA_C_16BF => rocblas_datatype::rocblas_datatype_bf16_c,
        _ => panic!(),
    }
}

unsafe fn set_stream(handle: cublasHandle_t, stream_id: cudaStream_t) -> cublasStatus_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_get_export_table = lib
        .get::<unsafe extern "C" fn(
            ppExportTable: *mut *const ::std::os::raw::c_void,
            pExportTableId: *const CUuuid,
        ) -> CUresult>(b"cuGetExportTable\0")
        .unwrap();
    let mut export_table = ptr::null();
    (cu_get_export_table)(&mut export_table, &zluda_dark_api::ZludaExt::GUID);
    let zluda_ext = zluda_dark_api::ZludaExt::new(export_table);
    let stream: Result<_, _> = zluda_ext.get_hip_stream(stream_id as _).into();
    to_cuda(rocblas_set_stream(handle as _, stream.unwrap() as _))
}

unsafe fn get_stream(handle: cublasHandle_t, stream_id: *mut cudaStream_t) -> cublasStatus_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_get_export_table = lib
        .get::<unsafe extern "C" fn(
            ppExportTable: *mut *const ::std::os::raw::c_void,
            pExportTableId: *const CUuuid,
        ) -> CUresult>(b"cuGetExportTable\0")
        .unwrap();
    let mut export_table = ptr::null();
    (cu_get_export_table)(&mut export_table, &zluda_dark_api::ZludaExt::GUID);
    let zluda_ext = zluda_dark_api::ZludaExt::new(export_table);
    let stream: Result<_, _> = zluda_ext.get_hip_stream(stream_id as _).into();
    to_cuda(rocblas_get_stream(handle as _, stream.unwrap() as _))
}

fn set_math_mode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t {
    // llama.cpp uses CUBLAS_TF32_TENSOR_OP_MATH
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

unsafe fn sgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: *const f32,
    c: *mut f32,
    ldc: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_sgemm(
        handle as _,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    ))
}

unsafe fn init() -> cublasStatus_t {
    rocblas_initialize();
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

unsafe fn dasum_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *const f64,
    incx: i32,
    result: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_dasum(handle as _, n, x, incx, result))
}

unsafe fn daxpy_v2(
    handle: *mut cublasContext,
    n: i32,
    alpha: *const f64,
    x: *const f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) -> cublasStatus_t {
    to_cuda(rocblas_daxpy(handle as _, n, alpha, x, incx, y, incy))
}

unsafe fn ddot_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    result: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_ddot(handle as _, n, x, incx, y, incy, result))
}

unsafe fn dscal_v2(
    handle: *mut cublasContext,
    n: i32,
    alpha: *const f64,
    x: *mut f64,
    incx: i32,
) -> cublasStatus_t {
    to_cuda(rocblas_dscal(handle as _, n, alpha, x, incx))
}

unsafe fn dnrm_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *const f64,
    incx: i32,
    result: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_dnrm2(handle.cast(), n, x, incx, result))
}

unsafe fn idamax_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *const f64,
    incx: i32,
    result: *mut i32,
) -> cublasStatus_t {
    to_cuda(rocblas_idamax(handle.cast(), n, x, incx, result))
}

unsafe fn idamin_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *const f64,
    incx: i32,
    result: *mut i32,
) -> cublasStatus_t {
    to_cuda(rocblas_idamin(handle.cast(), n, x, incx, result))
}

unsafe fn set_workspace(
    handle: *mut cublasContext,
    workspace: *mut std::ffi::c_void,
    workspace_size_in_bytes: usize,
) -> cublasStatus_t {
    to_cuda(rocblas_set_workspace(
        handle.cast(),
        workspace,
        workspace_size_in_bytes,
    ))
}

unsafe fn gemm_ex(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    atype: cudaDataType_t,
    lda: i32,
    b: *const std::ffi::c_void,
    btype: cudaDataType_t,
    ldb: i32,
    beta: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
    ctype: cudaDataType_t,
    ldc: i32,
    compute_type: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    let atype = type_from_cuda(atype);
    let btype = type_from_cuda(btype);
    let ctype = type_from_cuda(ctype);
    let compute_type = to_compute_type(compute_type);
    let algo = to_algo(algo);
    to_cuda(rocblas_gemm_ex(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        b,
        btype,
        ldb,
        beta,
        c,
        ctype,
        ldc,
        c,
        ctype,
        ldc,
        compute_type,
        algo,
        0,
        0,
    ))
}

fn to_algo(algo: cublasGemmAlgo_t) -> rocblas_gemm_algo_ {
    // only option
    rocblas_gemm_algo::rocblas_gemm_algo_standard
}

fn to_compute_type(compute_type: cublasComputeType_t) -> rocblas_datatype {
    match compute_type {
        cublasComputeType_t::CUBLAS_COMPUTE_16F
        | cublasComputeType_t::CUBLAS_COMPUTE_16F_PEDANTIC
        | cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F => {
            rocblas_datatype::rocblas_datatype_f16_r
        }
        cublasComputeType_t::CUBLAS_COMPUTE_32F
        | cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
        | cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => {
            rocblas_datatype::rocblas_datatype_f32_r
        }
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF => {
            rocblas_datatype::rocblas_datatype_bf16_r
        }
        cublasComputeType_t::CUBLAS_COMPUTE_64F
        | cublasComputeType_t::CUBLAS_COMPUTE_64F_PEDANTIC => {
            rocblas_datatype::rocblas_datatype_f64_r
        }
        cublasComputeType_t::CUBLAS_COMPUTE_32I
        | cublasComputeType_t::CUBLAS_COMPUTE_32I_PEDANTIC => {
            rocblas_datatype::rocblas_datatype_i32_r
        }
        _ => panic!(),
    }
}

unsafe fn zgemm_strided_batch(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const double2,
    a: *const double2,
    lda: i32,
    stride_a: i64,
    b: *const double2,
    ldb: i32,
    stride_b: i64,
    beta: *const double2,
    c: *mut double2,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_zgemm_strided_batched(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a.cast(),
        lda,
        stride_a,
        b.cast(),
        ldb,
        stride_b,
        beta.cast(),
        c.cast(),
        ldc,
        stride_c,
        batch_count,
    ))
}

unsafe fn cgemm_strided_batch(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const float2,
    a: *const float2,
    lda: i32,
    stride_a: i64,
    b: *const float2,
    ldb: i32,
    stride_b: i64,
    beta: *const float2,
    c: *mut float2,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_cgemm_strided_batched(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a.cast(),
        lda,
        stride_a,
        b.cast(),
        ldb,
        stride_b,
        beta.cast(),
        c.cast(),
        ldc,
        stride_c,
        batch_count,
    ))
}

unsafe fn dgemm_strided_batch(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    stride_a: i64,
    b: *const f64,
    ldb: i32,
    stride_b: i64,
    beta: *const f64,
    c: *mut f64,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_dgemm_strided_batched(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        stride_a,
        b,
        ldb,
        stride_b,
        beta,
        c,
        ldc,
        stride_c,
        batch_count,
    ))
}

unsafe fn sgemm_strided_batch(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    stride_a: i64,
    b: *const f32,
    ldb: i32,
    stride_b: i64,
    beta: *const f32,
    c: *mut f32,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_sgemm_strided_batched(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        stride_a,
        b,
        ldb,
        stride_b,
        beta,
        c,
        ldc,
        stride_c,
        batch_count,
    ))
}

unsafe fn zgetrf_batched(
    handle: *mut cublasContext,
    n: i32,
    a: *const *mut double2,
    lda: i32,
    p: *mut i32,
    info: *mut i32,
    batch_size: i32,
) -> cublasStatus_t {
    to_cuda_solver(rocsolver_zgetrf_batched(
        handle.cast(),
        n,
        n,
        a.cast(),
        lda,
        p,
        n as i64,
        info,
        batch_size,
    ))
}

unsafe fn cgetrf_batched(
    handle: *mut cublasContext,
    n: i32,
    a: *const *mut float2,
    lda: i32,
    p: *mut i32,
    info: *mut i32,
    batch_size: i32,
) -> cublasStatus_t {
    to_cuda_solver(rocsolver_cgetrf_batched(
        handle.cast(),
        n,
        n,
        a.cast(),
        lda,
        p,
        n as i64,
        info,
        batch_size,
    ))
}

unsafe fn zgetri_batched(
    handle: *mut cublasContext,
    n: i32,
    a: *const *const double2,
    lda: i32,
    p: *const i32,
    c: *const *mut double2,
    ldc: i32,
    info: *mut i32,
    batch_size: i32,
) -> cublasStatus_t {
    to_cuda_solver(rocsolver_zgetri_outofplace_batched(
        handle.cast(),
        n,
        a.cast(),
        lda,
        p.cast_mut(),
        n as i64,
        c.cast(),
        ldc,
        info,
        batch_size,
    ))
}

unsafe fn cgetri_batched(
    handle: *mut cublasContext,
    n: i32,
    a: *const *const float2,
    lda: i32,
    p: *const i32,
    c: *const *mut float2,
    ldc: i32,
    info: *mut i32,
    batch_size: i32,
) -> cublasStatus_t {
    to_cuda_solver(rocsolver_cgetri_outofplace_batched(
        handle.cast(),
        n,
        a.cast(),
        lda,
        p.cast_mut(),
        n as i64,
        c.cast(),
        ldc,
        info,
        batch_size,
    ))
}

unsafe fn dtrmm_v2(
    handle: *mut cublasContext,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    transa: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    c: *mut f64,
    ldc: i32,
) -> cublasStatus_t {
    let side = to_side(side);
    let uplo = to_fill(uplo);
    let transa = op_from_cuda(transa);
    let diag = to_diag(diag);
    to_cuda(rocblas_dtrmm_outofplace(
        handle.cast(),
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        c,
        ldc,
    ))
}

fn to_side(side: cublasSideMode_t) -> rocblas_side {
    match side {
        cublasSideMode_t::CUBLAS_SIDE_LEFT => rocblas_side::rocblas_side_left,
        cublasSideMode_t::CUBLAS_SIDE_RIGHT => rocblas_side::rocblas_side_right,
        _ => panic!(),
    }
}

fn to_fill(uplo: cublasFillMode_t) -> rocblas_fill {
    match uplo {
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER => rocblas_fill::rocblas_fill_lower,
        cublasFillMode_t::CUBLAS_FILL_MODE_UPPER => rocblas_fill::rocblas_fill_upper,
        cublasFillMode_t::CUBLAS_FILL_MODE_FULL => rocblas_fill::rocblas_fill_full,
        _ => panic!(),
    }
}

fn to_diag(diag: cublasDiagType_t) -> rocblas_diagonal {
    match diag {
        cublasDiagType_t::CUBLAS_DIAG_NON_UNIT => rocblas_diagonal::rocblas_diagonal_non_unit,
        cublasDiagType_t::CUBLAS_DIAG_UNIT => rocblas_diagonal::rocblas_diagonal_unit,
        _ => panic!(),
    }
}

unsafe fn dgemv_v2(
    handle: *mut cublasContext,
    trans: cublasOperation_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: *const f64,
    y: *mut f64,
    incy: i32,
) -> cublasStatus_t {
    let trans: rocblas_operation = op_from_cuda(trans);
    to_cuda(rocblas_dgemv(
        handle.cast(),
        trans,
        m,
        n,
        alpha,
        a,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn get_pointer_mode(
    handle: cublasHandle_t,
    mode: *mut cublasPointerMode_t,
) -> cublasStatus_t {
    to_cuda(rocblas_get_pointer_mode(handle.cast(), mode.cast()))
}

unsafe fn set_pointer_mode(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t {
    to_cuda(rocblas_set_pointer_mode(
        handle.cast(),
        rocblas_pointer_mode_(mode.0),
    ))
}

unsafe fn drot_v2(
    handle: *mut cublasContext,
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    c: *const f64,
    s: *const f64,
) -> cublasStatus_t {
    to_cuda(rocblas_drot(handle.cast(), n, x, incx, y, incy, c, s))
}

unsafe fn drotg(
    handle: *mut cublasContext,
    a: *mut f64,
    b: *mut f64,
    c: *mut f64,
    s: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_drotg(handle.cast(), a, b, c, s))
}

unsafe fn drotm(
    handle: *mut cublasContext,
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    param: *const f64,
) -> cublasStatus_t {
    to_cuda(rocblas_drotm(handle.cast(), n, x, incx, y, incy, param))
}

unsafe fn drotmg(
    handle: *mut cublasContext,
    d1: *mut f64,
    d2: *mut f64,
    x1: *mut f64,
    y1: *const f64,
    param: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_drotmg(handle.cast(), d1, d2, x1, y1, param))
}

unsafe fn dswap(
    handle: *mut cublasContext,
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) -> cublasStatus_t {
    to_cuda(rocblas_dswap(handle.cast(), n, x, incx, y, incy))
}

unsafe fn dger(
    handle: *mut cublasContext,
    m: i32,
    n: i32,
    alpha: *const f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) -> cublasStatus_t {
    to_cuda(rocblas_dger(
        handle.cast(),
        m,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        a,
        lda,
    ))
}

unsafe fn dgemm(
    handle: *mut cublasContext,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: *const f64,
    c: *mut f64,
    ldc: i32,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    to_cuda(rocblas_dgemm(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    ))
}

unsafe fn dtrsm(
    handle: *mut cublasContext,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i32,
    n: i32,
    alpha: *const f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) -> cublasStatus_t {
    let side = to_side(side);
    let uplo = to_fill(uplo);
    let trans = op_from_cuda(trans);
    let diag = to_diag(diag);
    to_cuda(rocblas_dtrsm(
        handle.cast(),
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
    ))
}

unsafe fn gemm_batched_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const std::ffi::c_void,
    a: *const *const std::ffi::c_void,
    atype: cudaDataType_t,
    lda: i32,
    b: *const *const std::ffi::c_void,
    btype: cudaDataType_t,
    ldb: i32,
    beta: *const std::ffi::c_void,
    c: *const *mut std::ffi::c_void,
    ctype: cudaDataType_t,
    ldc: i32,
    batch_count: i32,
    compute_type: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    let atype = type_from_cuda(atype);
    let btype = type_from_cuda(btype);
    let ctype = type_from_cuda(ctype);
    let compute_type = to_compute_type(compute_type);
    let algo = to_algo(algo);
    to_cuda(rocblas_gemm_batched_ex(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a.cast(),
        atype,
        lda,
        b.cast(),
        btype,
        ldb,
        beta,
        c.cast(),
        ctype,
        ldc,
        c.cast_mut().cast(),
        ctype,
        ldc,
        batch_count,
        compute_type,
        algo,
        0,
        rocblas_gemm_flags::rocblas_gemm_flags_none.0,
    ))
}

unsafe fn gemm_strided_batched_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const ::std::os::raw::c_void,
    a: *const ::std::os::raw::c_void,
    atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    stride_a: ::std::os::raw::c_longlong,
    b: *const ::std::os::raw::c_void,
    btype: cudaDataType,
    ldb: ::std::os::raw::c_int,
    stride_b: ::std::os::raw::c_longlong,
    beta: *const ::std::os::raw::c_void,
    c: *mut ::std::os::raw::c_void,
    ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
    stride_c: ::std::os::raw::c_longlong,
    batch_count: ::std::os::raw::c_int,
    compute_type: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    let transa = op_from_cuda(transa);
    let transb = op_from_cuda(transb);
    let atype = type_from_cuda(atype);
    let btype = type_from_cuda(btype);
    let ctype = type_from_cuda(ctype);
    let compute_type = to_compute_type(compute_type);
    let algo = to_algo(algo);
    to_cuda(rocblas_gemm_strided_batched_ex(
        handle.cast(),
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        atype,
        lda,
        stride_a,
        b,
        btype,
        ldb,
        stride_b,
        beta,
        c,
        ctype,
        ldc,
        stride_c,
        c,
        ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
        0,
        rocblas_gemm_flags::rocblas_gemm_flags_none.0,
    ))
}

unsafe fn get_atomics_mode(
    handle: cublasHandle_t,
    mode: *mut cublasAtomicsMode_t,
) -> cublasStatus_t {
    to_cuda(rocblas_get_atomics_mode(handle.cast(), mode.cast()))
}

unsafe fn set_atomics_mode(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t {
    to_cuda(rocblas_set_atomics_mode(
        handle.cast(),
        rocblas_atomics_mode(mode.0),
    ))
}

unsafe fn sgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    beta: *const f32,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a: rocblas_operation_ = op_from_cuda(transa);
    let op_b: rocblas_operation_ = op_from_cuda(transb);

    to_cuda(rocblas_sgeam(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        alpha,
        A,
        lda,
        beta,
        B,
        ldb,
        C,
        ldc,
    ))
}

unsafe fn dcopy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_dcopy(handle.cast(), n, x, incx, y, incy))
}

unsafe fn zdotc_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    result: *mut cuDoubleComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_zdotc(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        result.cast(),
    ))
}

unsafe fn dgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    kl: ::std::os::raw::c_int,
    ku: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    beta: *const f64,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dgbmv(
        handle.cast(),
        op,
        m,
        n,
        kl,
        ku,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn zhbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zhbmv(
        handle.cast(),
        fillMode,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zhemv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zhemv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zher_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zher(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x.cast(),
        incx,
        A.cast(),
        lda,
    ))
}

unsafe fn zher2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zher2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn zhpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    AP: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zhpmv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        AP.cast(),
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zhpr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    AP: *mut cuDoubleComplex,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zhpr(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x.cast(),
        incx,
        AP.cast(),
    ))
}

unsafe fn zhpr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    AP: *mut cuDoubleComplex,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zhpr2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        AP.cast(),
    ))
}

pub unsafe extern "system" fn dsbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    beta: *const f64,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dsbmv(
        handle.cast(),
        fillMode,
        n,
        k,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn dspmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    AP: *const f64,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    beta: *const f64,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dspmv(
        handle.cast(),
        fillMode,
        n,
        alpha,
        AP,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn dspr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    AP: *mut f64,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dspr(handle.cast(), fillMode, n, alpha, x, incx, AP))
}

unsafe fn dspr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    y: *const f64,
    incy: ::std::os::raw::c_int,
    AP: *mut f64,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dspr2(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        AP,
    ))
}

unsafe fn dsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    beta: *const f64,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dsymv(
        handle.cast(),
        fillMode,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn set_vector(
    n: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    incx: ::std::os::raw::c_int,
    devicePtr: *mut ::std::os::raw::c_void,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_set_vector(n, elemSize, x, incx, devicePtr, incy))
}

unsafe fn get_vector(
    n: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_get_vector(n, elemSize, x, incx, y, incy))
}

unsafe fn set_matrix(
    rows: ::std::os::raw::c_int,
    cols: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    A: *const ::std::os::raw::c_void,
    lda: ::std::os::raw::c_int,
    B: *mut ::std::os::raw::c_void,
    ldb: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_set_matrix(rows, cols, elemSize, A, lda, B, ldb))
}

unsafe fn get_matrix(
    rows: ::std::os::raw::c_int,
    cols: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    A: *const ::std::os::raw::c_void,
    lda: ::std::os::raw::c_int,
    B: *mut ::std::os::raw::c_void,
    ldb: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_get_matrix(rows, cols, elemSize, A, lda, B, ldb))
}

unsafe fn set_vector_async(
    n: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    hostPtr: *const ::std::os::raw::c_void,
    incx: ::std::os::raw::c_int,
    devicePtr: *mut ::std::os::raw::c_void,
    incy: ::std::os::raw::c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    to_cuda(rocblas_set_vector_async(
        n,
        elemSize,
        hostPtr,
        incx,
        devicePtr,
        incy,
        stream.cast(),
    ))
}

unsafe fn get_vector_async(
    n: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    devicePtr: *const ::std::os::raw::c_void,
    incx: ::std::os::raw::c_int,
    hostPtr: *mut ::std::os::raw::c_void,
    incy: ::std::os::raw::c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    to_cuda(rocblas_get_vector_async(
        n,
        elemSize,
        devicePtr,
        incx,
        hostPtr,
        incy,
        stream.cast(),
    ))
}

unsafe fn set_matrix_async(
    rows: ::std::os::raw::c_int,
    cols: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    A: *const ::std::os::raw::c_void,
    lda: ::std::os::raw::c_int,
    B: *mut ::std::os::raw::c_void,
    ldb: ::std::os::raw::c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    to_cuda(rocblas_set_matrix_async(
        rows,
        cols,
        elemSize,
        A,
        lda,
        B,
        ldb,
        stream.cast(),
    ))
}

unsafe fn get_matrix_async(
    rows: ::std::os::raw::c_int,
    cols: ::std::os::raw::c_int,
    elemSize: ::std::os::raw::c_int,
    A: *const ::std::os::raw::c_void,
    lda: ::std::os::raw::c_int,
    B: *mut ::std::os::raw::c_void,
    ldb: ::std::os::raw::c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    to_cuda(rocblas_get_matrix_async(
        rows,
        cols,
        elemSize,
        A,
        lda,
        B,
        ldb,
        stream.cast(),
    ))
}

unsafe fn nrm2_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
) -> cublasStatus_t {
    let x_type = type_from_cuda(xType);
    let result_type = type_from_cuda(resultType);
    let execution_type = type_from_cuda(executionType);
    to_cuda(rocblas_nrm2_ex(
        handle.cast(),
        n,
        x,
        x_type,
        incx,
        result,
        result_type,
        execution_type,
    ))
}

unsafe fn snrm2_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_snrm2(handle.cast(), n, x, incx, result))
}

unsafe fn scnrm2_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_scnrm2(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn dznrm2_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    result: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_dznrm2(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn dot_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *const ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
) -> cublasStatus_t {
    let x_type = type_from_cuda(xType);
    let y_type = type_from_cuda(yType);
    let result_type = type_from_cuda(resultType);
    let execution_type = type_from_cuda(executionType);
    to_cuda(rocblas_dot_ex(
        handle.cast(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    ))
}

unsafe fn dotc_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *const ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
) -> cublasStatus_t {
    let x_type = type_from_cuda(xType);
    let y_type = type_from_cuda(yType);
    let result_type = type_from_cuda(resultType);
    let execution_type = type_from_cuda(executionType);
    to_cuda(rocblas_dotc_ex(
        handle.cast(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    ))
}

unsafe fn sdot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *const f32,
    incy: ::std::os::raw::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_sdot(handle.cast(), n, x, incx, y, incy, result))
}

unsafe fn cdotu_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    result: *mut cuComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_cdotu(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        result.cast(),
    ))
}

unsafe fn cdotc_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    result: *mut cuComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_cdotc(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        result.cast(),
    ))
}

unsafe fn zdotu_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    result: *mut cuDoubleComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_zdotu(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        result.cast(),
    ))
}

unsafe fn scal_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const ::std::os::raw::c_void,
    alphaType: cudaDataType,
    x: *mut ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    executionType: cudaDataType,
) -> cublasStatus_t {
    let alpha_type = type_from_cuda(alphaType);
    let x_type = type_from_cuda(xType);
    let execution_type = type_from_cuda(executionType);
    to_cuda(rocblas_scal_ex(
        handle.cast(),
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        execution_type,
    ))
}

unsafe fn sscal_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_sscal(handle.cast(), n, alpha, x, incx))
}

unsafe fn cscal_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_cscal(
        handle.cast(),
        n,
        alpha.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn csscal_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_csscal(handle.cast(), n, alpha, x.cast(), incx))
}

unsafe fn zscal_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zscal(
        handle.cast(),
        n,
        alpha.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn zdscal_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zdscal(handle.cast(), n, alpha, x.cast(), incx))
}

unsafe fn axpy_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const ::std::os::raw::c_void,
    alphaType: cudaDataType,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    let alpha_type = type_from_cuda(alphaType);
    let x_type = type_from_cuda(xType);
    let y_type = type_from_cuda(yType);
    let execution_type = type_from_cuda(executiontype);
    to_cuda(rocblas_axpy_ex(
        handle.cast(),
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        execution_type,
    ))
}

unsafe fn saxpy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_saxpy(handle.cast(), n, alpha, x, incx, y, incy))
}

unsafe fn caxpy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_caxpy(
        handle.cast(),
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn zaxpy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zaxpy(
        handle.cast(),
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn copy_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn scopy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_scopy(handle.cast(), n, x, incx, y, incy))
}

unsafe fn ccopy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_ccopy(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn zcopy_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zcopy(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn sswap_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_sswap(handle.cast(), n, x, incx, y, incy))
}

unsafe fn cswap_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_cswap(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn zswap_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zswap(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
    ))
}

unsafe fn swap_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn isamax_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_isamax(handle.cast(), n, x, incx, result))
}

unsafe fn icamax_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_icamax(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn izamax_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_izamax(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn iamax_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn isamin_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_isamin(handle.cast(), n, x, incx, result))
}

unsafe fn icamin_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_icamin(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn izamin_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_izamin(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn iamin_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn asum_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    result: *mut ::std::os::raw::c_void,
    resultType: cudaDataType,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn sasum_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_sasum(handle.cast(), n, x, incx, result))
}

unsafe fn scasum_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_scasum(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn dzasum_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    result: *mut f64,
) -> cublasStatus_t {
    to_cuda(rocblas_dzasum(handle.cast(), n, x.cast(), incx, result))
}

unsafe fn srot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
    c: *const f32,
    s: *const f32,
) -> cublasStatus_t {
    to_cuda(rocblas_srot(handle.cast(), n, x, incx, y, incy, c, s))
}

unsafe fn crot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
    c: *const f32,
    s: *const cuComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_crot(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        c,
        s.cast(),
    ))
}

unsafe fn csrot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
    c: *const f32,
    s: *const f32,
) -> cublasStatus_t {
    to_cuda(rocblas_csrot(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        c,
        s,
    ))
}

unsafe fn zrot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    c: *const f64,
    s: *const cuDoubleComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_zrot(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        c,
        s.cast(),
    ))
}

unsafe fn zdrot_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    c: *const f64,
    s: *const f64,
) -> cublasStatus_t {
    to_cuda(rocblas_zdrot(
        handle.cast(),
        n,
        x.cast(),
        incx,
        y.cast(),
        incy,
        c,
        s,
    ))
}

unsafe fn rot_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
    c: *const ::std::os::raw::c_void,
    s: *const ::std::os::raw::c_void,
    csType: cudaDataType,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    let x_type = type_from_cuda(xType);
    let y_type = type_from_cuda(yType);
    let cs_type = type_from_cuda(csType);
    let execution_type = type_from_cuda(executiontype);
    to_cuda(rocblas_rot_ex(
        handle.cast(),
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c,
        s,
        cs_type,
        execution_type,
    ))
}

unsafe fn srotg_v2(
    handle: cublasHandle_t,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    s: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_srotg(handle.cast(), a, b, c, s))
}

unsafe fn crotg_v2(
    handle: cublasHandle_t,
    a: *mut cuComplex,
    b: *mut cuComplex,
    c: *mut f32,
    s: *mut cuComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_crotg(
        handle.cast(),
        a.cast(),
        b.cast(),
        c,
        s.cast(),
    ))
}

unsafe fn zrotg_v2(
    handle: cublasHandle_t,
    a: *mut cuDoubleComplex,
    b: *mut cuDoubleComplex,
    c: *mut f64,
    s: *mut cuDoubleComplex,
) -> cublasStatus_t {
    to_cuda(rocblas_zrotg(
        handle.cast(),
        a.cast(),
        b.cast(),
        c,
        s.cast(),
    ))
}

unsafe fn rotg_ex(
    handle: cublasHandle_t,
    a: *mut ::std::os::raw::c_void,
    b: *mut ::std::os::raw::c_void,
    abType: cudaDataType,
    c: *mut ::std::os::raw::c_void,
    s: *mut ::std::os::raw::c_void,
    csType: cudaDataType,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn srotm_v2(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
    param: *const f32,
) -> cublasStatus_t {
    to_cuda(rocblas_srotm(handle.cast(), n, x, incx, y, incy, param))
}

unsafe fn rotm_ex(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    x: *mut ::std::os::raw::c_void,
    xType: cudaDataType,
    incx: ::std::os::raw::c_int,
    y: *mut ::std::os::raw::c_void,
    yType: cudaDataType,
    incy: ::std::os::raw::c_int,
    param: *const ::std::os::raw::c_void,
    paramType: cudaDataType,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn srotmg_v2(
    handle: cublasHandle_t,
    d1: *mut f32,
    d2: *mut f32,
    x1: *mut f32,
    y1: *const f32,
    param: *mut f32,
) -> cublasStatus_t {
    to_cuda(rocblas_srotmg(handle.cast(), d1, d2, x1, y1, param))
}

unsafe fn rotmg_ex(
    handle: cublasHandle_t,
    d1: *mut ::std::os::raw::c_void,
    d1Type: cudaDataType,
    d2: *mut ::std::os::raw::c_void,
    d2Type: cudaDataType,
    x1: *mut ::std::os::raw::c_void,
    x1Type: cudaDataType,
    y1: *const ::std::os::raw::c_void,
    y1Type: cudaDataType,
    param: *mut ::std::os::raw::c_void,
    paramType: cudaDataType,
    executiontype: cudaDataType,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn cgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cgemv(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zgemv(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn sgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    kl: ::std::os::raw::c_int,
    ku: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    beta: *const f32,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_sgbmv(
        handle.cast(),
        op,
        m,
        n,
        kl,
        ku,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn cgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    kl: ::std::os::raw::c_int,
    ku: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cgbmv(
        handle.cast(),
        op,
        m,
        n,
        kl,
        ku,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    kl: ::std::os::raw::c_int,
    ku: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zgbmv(
        handle.cast(),
        op,
        m,
        n,
        kl,
        ku,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn strmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_strmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn dtrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtrmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn ctrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctrmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn ztrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztrmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn stbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_stbmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn dtbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtbmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn ctbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctbmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn ztbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztbmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn stpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const f32,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_stpmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP,
        x,
        incx,
    ))
}

unsafe fn dtpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const f64,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtpmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP,
        x,
        incx,
    ))
}

unsafe fn ctpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const cuComplex,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctpmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn ztpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztpmv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn strsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_strsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn dtrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtrsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn ctrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctrsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn ztrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztrsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn stpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const f32,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_stpsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP,
        x,
        incx,
    ))
}

unsafe fn dtpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const f64,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtpsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP,
        x,
        incx,
    ))
}

unsafe fn ctpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const cuComplex,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctpsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn ztpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    AP: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztpsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        AP.cast(),
        x.cast(),
        incx,
    ))
}

unsafe fn stbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *mut f32,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_stbsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn dtbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *mut f64,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_dtbsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A,
        lda,
        x,
        incx,
    ))
}

unsafe fn ctbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ctbsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn ztbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *mut cuDoubleComplex,
    incx: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diagType = to_diag(diag);
    to_cuda(rocblas_ztbsv(
        handle.cast(),
        fillMode,
        op,
        diagType,
        n,
        k,
        A.cast(),
        lda,
        x.cast(),
        incx,
    ))
}

unsafe fn ssymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    beta: *const f32,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_ssymv(
        handle.cast(),
        fillMode,
        n,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn csymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_csymv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn zsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zsymv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn chemv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_chemv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn ssbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    beta: *const f32,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_ssbmv(
        handle.cast(),
        fillMode,
        n,
        k,
        alpha,
        A,
        lda,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn chbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_chbmv(
        handle.cast(),
        fillMode,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn sspmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    AP: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    beta: *const f32,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_sspmv(
        handle.cast(),
        fillMode,
        n,
        alpha,
        AP,
        x,
        incx,
        beta,
        y,
        incy,
    ))
}

unsafe fn chpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    AP: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_chpmv(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        AP.cast(),
        x.cast(),
        incx,
        beta.cast(),
        y.cast(),
        incy,
    ))
}

unsafe fn sger_v2(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *const f32,
    incy: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_sger(
        handle.cast(),
        m,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        A,
        lda,
    ))
}

unsafe fn cgeru_v2(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_cgeru(
        handle.cast(),
        m,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn cgerc_v2(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_cgerc(
        handle.cast(),
        m,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn zgeru_v2(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zgeru(
        handle.cast(),
        m,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn zgerc_v2(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    to_cuda(rocblas_zgerc(
        handle.cast(),
        m,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn ssyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_ssyr(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        A,
        lda,
    ))
}

unsafe fn dsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dsyr(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        A,
        lda,
    ))
}

unsafe fn csyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_csyr(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        A.cast(),
        lda,
    ))
}

unsafe fn zsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zsyr(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        A.cast(),
        lda,
    ))
}

unsafe fn cher_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_cher(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x.cast(),
        incx,
        A.cast(),
        lda,
    ))
}

unsafe fn sspr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    AP: *mut f32,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_sspr(handle.cast(), fillMode, n, alpha, x, incx, AP))
}

unsafe fn chpr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    AP: *mut cuComplex,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_chpr(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x.cast(),
        incx,
        AP.cast(),
    ))
}

unsafe fn ssyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *const f32,
    incy: ::std::os::raw::c_int,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_ssyr2(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        A,
        lda,
    ))
}

unsafe fn dsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    y: *const f64,
    incy: ::std::os::raw::c_int,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_dsyr2(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        A,
        lda,
    ))
}

unsafe fn csyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_csyr2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn zsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_zsyr2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn cher2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_cher2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        A.cast(),
        lda,
    ))
}

unsafe fn sspr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    y: *const f32,
    incy: ::std::os::raw::c_int,
    AP: *mut f32,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_sspr2(
        handle.cast(),
        fillMode,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        AP,
    ))
}

unsafe fn chpr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    y: *const cuComplex,
    incy: ::std::os::raw::c_int,
    AP: *mut cuComplex,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    to_cuda(rocblas_chpr2(
        handle.cast(),
        fillMode,
        n,
        alpha.cast(),
        x.cast(),
        incx,
        y.cast(),
        incy,
        AP.cast(),
    ))
}

unsafe fn sgemv_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    Aarray: *const *const f32,
    lda: ::std::os::raw::c_int,
    xarray: *const *const f32,
    incx: ::std::os::raw::c_int,
    beta: *const f32,
    yarray: *const *mut f32,
    incy: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_sgemv_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha,
        Aarray,
        lda,
        xarray,
        incx,
        beta,
        yarray,
        incy,
        batchCount,
    ))
}

unsafe fn dgemv_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    Aarray: *const *const f64,
    lda: ::std::os::raw::c_int,
    xarray: *const *const f64,
    incx: ::std::os::raw::c_int,
    beta: *const f64,
    yarray: *const *mut f64,
    incy: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dgemv_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha,
        Aarray,
        lda,
        xarray,
        incx,
        beta,
        yarray,
        incy,
        batchCount,
    ))
}

unsafe fn cgemv_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    Aarray: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    xarray: *const *const cuComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuComplex,
    yarray: *const *mut cuComplex,
    incy: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cgemv_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        Aarray.cast(),
        lda,
        xarray.cast(),
        incx,
        beta.cast(),
        yarray.cast(),
        incy,
        batchCount,
    ))
}

unsafe fn zgemv_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    Aarray: *const *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    xarray: *const *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    yarray: *const *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zgemv_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        Aarray.cast(),
        lda,
        xarray.cast(),
        incx,
        beta.cast(),
        yarray.cast(),
        incy,
        batchCount,
    ))
}

unsafe fn sgemv_strided_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    stridex: ::std::os::raw::c_longlong,
    beta: *const f32,
    y: *mut f32,
    incy: ::std::os::raw::c_int,
    stridey: ::std::os::raw::c_longlong,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_sgemv_strided_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha,
        A,
        lda,
        strideA,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batchCount,
    ))
}

unsafe fn dgemv_strided_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    stridex: ::std::os::raw::c_longlong,
    beta: *const f64,
    y: *mut f64,
    incy: ::std::os::raw::c_int,
    stridey: ::std::os::raw::c_longlong,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dgemv_strided_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha,
        A,
        lda,
        strideA,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batchCount,
    ))
}

unsafe fn cgemv_strided_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    stridex: ::std::os::raw::c_longlong,
    beta: *const cuComplex,
    y: *mut cuComplex,
    incy: ::std::os::raw::c_int,
    stridey: ::std::os::raw::c_longlong,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cgemv_strided_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        strideA,
        x.cast(),
        incx,
        stridex,
        beta.cast(),
        y.cast(),
        incy,
        stridey,
        batchCount,
    ))
}

unsafe fn zgemv_strided_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    stridex: ::std::os::raw::c_longlong,
    beta: *const cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: ::std::os::raw::c_int,
    stridey: ::std::os::raw::c_longlong,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zgemv_strided_batched(
        handle.cast(),
        op,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        strideA,
        x.cast(),
        incx,
        stridex,
        beta.cast(),
        y.cast(),
        incy,
        stridey,
        batchCount,
    ))
}

unsafe fn cgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_cgemm(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn cgemm3m(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn cgemm3m_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    B: *const ::std::os::raw::c_void,
    Btype: cudaDataType,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn zgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_zgemm(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zgemm3m(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}
unsafe fn cgemm_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    B: *const ::std::os::raw::c_void,
    Btype: cudaDataType,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn uint8gemm_bias(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    transc: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    A: *const ::std::os::raw::c_uchar,
    A_bias: ::std::os::raw::c_int,
    lda: ::std::os::raw::c_int,
    B: *const ::std::os::raw::c_uchar,
    B_bias: ::std::os::raw::c_int,
    ldb: ::std::os::raw::c_int,
    C: *mut ::std::os::raw::c_uchar,
    C_bias: ::std::os::raw::c_int,
    ldc: ::std::os::raw::c_int,
    C_mult: ::std::os::raw::c_int,
    C_shift: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ssyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_ssyrk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        beta,
        C,
        ldc,
    ))
}

unsafe fn dsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dsyrk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        beta,
        C,
        ldc,
    ))
}

unsafe fn csyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_csyrk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zsyrk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn csyrk_ex(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn csyrk3m_ex(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn cherk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cherk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A.cast(),
        lda,
        beta,
        C.cast(),
        ldc,
    ))
}

unsafe fn zherk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zherk(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A.cast(),
        lda,
        beta,
        C.cast(),
        ldc,
    ))
}

unsafe fn cherk_ex(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn cherk3m_ex(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const ::std::os::raw::c_void,
    Atype: cudaDataType,
    lda: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut ::std::os::raw::c_void,
    Ctype: cudaDataType,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ssyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_ssyr2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn dsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dsyr2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn csyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_csyr2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zsyr2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn cher2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cher2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta,
        C.cast(),
        ldc,
    ))
}

unsafe fn zher2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zher2k(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta,
        C.cast(),
        ldc,
    ))
}

unsafe fn ssyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_ssyrkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn dsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_dsyrkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn csyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_csyrkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zsyrkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn cherkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_cherkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zherkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    to_cuda(rocblas_zherkx(
        handle.cast(),
        fillMode,
        op,
        n,
        k,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn ssymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_ssymm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn dsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_dsymm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    ))
}

unsafe fn csymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_csymm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_zsymm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn chemm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_chemm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn zhemm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    to_cuda(rocblas_zhemm(
        handle.cast(),
        side_mode,
        fillMode,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        beta.cast(),
        C.cast(),
        ldc,
    ))
}

unsafe fn strsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *mut f32,
    ldb: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_strsm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha,
        A,
        lda,
        B,
        ldb,
    ))
}

unsafe fn ctrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuComplex,
    ldb: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ctrsm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
    ))
}

unsafe fn ztrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ztrsm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
    ))
}

unsafe fn strmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    B: *const f32,
    ldb: ::std::os::raw::c_int,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_strmm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha,
        A,
        lda,
        B as *mut f32,
        ldb,
    ))
}

unsafe fn ctrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ctrmm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        (B as *mut cuComplex).cast(),
        ldb,
    ))
}

unsafe fn ztrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let fillMode = to_fill(uplo);
    let side_mode = to_side(side);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ztrmm(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        (B as *mut cuDoubleComplex).cast(),
        ldb,
    ))
}

unsafe fn sgemm_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f32,
    Aarray: *const *const f32,
    lda: ::std::os::raw::c_int,
    Barray: *const *const f32,
    ldb: ::std::os::raw::c_int,
    beta: *const f32,
    Carray: *const *mut f32,
    ldc: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_sgemm_batched(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha,
        Aarray,
        lda,
        Barray,
        ldb,
        beta,
        Carray,
        ldc,
        batchCount,
    ))
}

unsafe fn dgemm_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const f64,
    Aarray: *const *const f64,
    lda: ::std::os::raw::c_int,
    Barray: *const *const f64,
    ldb: ::std::os::raw::c_int,
    beta: *const f64,
    Carray: *const *mut f64,
    ldc: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_dgemm_batched(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha,
        Aarray,
        lda,
        Barray,
        ldb,
        beta,
        Carray,
        ldc,
        batchCount,
    ))
}

unsafe fn cgemm_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    Aarray: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    Barray: *const *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    Carray: *const *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_cgemm_batched(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha.cast(),
        Aarray.cast(),
        lda,
        Barray.cast(),
        ldb,
        beta.cast(),
        Carray.cast(),
        ldc,
        batchCount,
    ))
}

unsafe fn cgemm3m_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    Aarray: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    Barray: *const *const cuComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuComplex,
    Carray: *const *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn zgemm_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    Aarray: *const *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Barray: *const *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    Carray: *const *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_zgemm_batched(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        k,
        alpha.cast(),
        Aarray.cast(),
        lda,
        Barray.cast(),
        ldb,
        beta.cast(),
        Carray.cast(),
        ldc,
        batchCount,
    ))
}

unsafe fn cgemm3m_strided_batched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    k: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    strideA: ::std::os::raw::c_longlong,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    strideB: ::std::os::raw::c_longlong,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    strideC: ::std::os::raw::c_longlong,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn dgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    beta: *const f64,
    B: *const f64,
    ldb: ::std::os::raw::c_int,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_dgeam(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        alpha,
        A,
        lda,
        beta,
        B,
        ldb,
        C,
        ldc,
    ))
}

unsafe fn cgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    beta: *const cuComplex,
    B: *const cuComplex,
    ldb: ::std::os::raw::c_int,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_cgeam(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        beta.cast(),
        B.cast(),
        ldb,
        C.cast(),
        ldc,
    ))
}

unsafe fn zgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    beta: *const cuDoubleComplex,
    B: *const cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let op_a = op_from_cuda(transa);
    let op_b = op_from_cuda(transb);
    to_cuda(rocblas_zgeam(
        handle.cast(),
        op_a,
        op_b,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        beta.cast(),
        B.cast(),
        ldb,
        C.cast(),
        ldc,
    ))
}

unsafe fn sgetrf_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *mut f32,
    lda: ::std::os::raw::c_int,
    P: *mut ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if !P.is_null() {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_sgetrf_batched(
            handle.cast(),
            n,
            n,
            A,
            lda,
            P,
            n.into(),
            info,
            batchSize,
        ))
    } else {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_sgetrf_npvt_batched(
            handle.cast(),
            n,
            n,
            A,
            lda,
            info,
            batchSize,
        ))
    }
}

unsafe fn dgetrf_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *mut f64,
    lda: ::std::os::raw::c_int,
    P: *mut ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if !P.is_null() {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_dgetrf_batched(
            handle.cast(),
            n,
            n,
            A,
            lda,
            P,
            n.into(),
            info,
            batchSize,
        ))
    } else {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_dgetrf_npvt_batched(
            handle.cast(),
            n,
            n,
            A,
            lda,
            info,
            batchSize,
        ))
    }
}

unsafe fn sgetri_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const f32,
    lda: ::std::os::raw::c_int,
    P: *const ::std::os::raw::c_int,
    C: *const *mut f32,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if !P.is_null() {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_sgetri_outofplace_batched(
            handle.cast(),
            n,
            A as *mut *mut f32,
            lda,
            P as *mut ::std::os::raw::c_int,
            n.into(),
            C,
            ldc,
            info,
            batchSize,
        ))
    } else {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_sgetri_npvt_outofplace_batched(
            handle.cast(),
            n,
            A as *mut *mut f32,
            lda,
            C,
            n,
            info,
            batchSize,
        ))
    }
}

unsafe fn dgetri_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const f64,
    lda: ::std::os::raw::c_int,
    P: *const ::std::os::raw::c_int,
    C: *const *mut f64,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if !P.is_null() {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_dgetri_outofplace_batched(
            handle.cast(),
            n,
            A as *mut *mut f64,
            lda,
            P as *mut ::std::os::raw::c_int,
            n.into(),
            C,
            ldc,
            info,
            batchSize,
        ))
    } else {
        rocsolver_to_cuda(rocsolver_sys::rocsolver_dgetri_npvt_outofplace_batched(
            handle.cast(),
            n,
            A as *mut *mut f64,
            lda,
            C,
            n,
            info,
            batchSize,
        ))
    }
}
unsafe fn sgetrs_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *const f32,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    Barray: *const *mut f32,
    ldb: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if n > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N
        && trans != cublasOperation_t::CUBLAS_OP_T
        && trans != cublasOperation_t::CUBLAS_OP_C
    {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if Aarray.is_null() && n != 0 {
        *info = -4;
    } else if lda < max {
        *info = -5;
    } else if devIpiv.is_null() && n != 0 {
        *info = -6;
    } else if Barray.is_null() && n * nrhs != 0 {
        *info = -7;
    } else if ldb < max {
        *info = -8;
    } else if batchSize < 0 {
        *info = -10
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_sgetrs_batched(
        handle.cast(),
        op,
        n,
        nrhs,
        Aarray as *mut *mut f32,
        lda,
        devIpiv,
        n.into(),
        Barray,
        ldb,
        batchSize,
    ))
}

unsafe fn dgetrs_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *const f64,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    Barray: *const *mut f64,
    ldb: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if n > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N
        && trans != cublasOperation_t::CUBLAS_OP_T
        && trans != cublasOperation_t::CUBLAS_OP_C
    {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if Aarray.is_null() && n != 0 {
        *info = -4;
    } else if lda < max {
        *info = -5;
    } else if devIpiv.is_null() && n != 0 {
        *info = -6;
    } else if Barray.is_null() && n * nrhs != 0 {
        *info = -7;
    } else if ldb < max {
        *info = -8;
    } else if batchSize < 0 {
        *info = -10
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_dgetrs_batched(
        handle.cast(),
        op,
        n,
        nrhs,
        Aarray as *mut *mut f64,
        lda,
        devIpiv,
        n.into(),
        Barray,
        ldb,
        batchSize,
    ))
}

unsafe fn cgetrs_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    Barray: *const *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if n > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N
        && trans != cublasOperation_t::CUBLAS_OP_T
        && trans != cublasOperation_t::CUBLAS_OP_C
    {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if Aarray.is_null() && n != 0 {
        *info = -4;
    } else if lda < max {
        *info = -5;
    } else if devIpiv.is_null() && n != 0 {
        *info = -6;
    } else if Barray.is_null() && n * nrhs != 0 {
        *info = -7;
    } else if ldb < max {
        *info = -8;
    } else if batchSize < 0 {
        *info = -10
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_cgetrs_batched(
        handle.cast(),
        op,
        n,
        nrhs,
        Aarray.cast(),
        lda,
        devIpiv,
        n.into(),
        Barray.cast(),
        ldb,
        batchSize,
    ))
}

unsafe fn zgetrs_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    devIpiv: *const ::std::os::raw::c_int,
    Barray: *const *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if n > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N
        && trans != cublasOperation_t::CUBLAS_OP_T
        && trans != cublasOperation_t::CUBLAS_OP_C
    {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if nrhs < 0 {
        *info = -3;
    } else if Aarray.is_null() && n != 0 {
        *info = -4;
    } else if lda < max {
        *info = -5;
    } else if devIpiv.is_null() && n != 0 {
        *info = -6;
    } else if Barray.is_null() && n * nrhs != 0 {
        *info = -7;
    } else if ldb < max {
        *info = -8;
    } else if batchSize < 0 {
        *info = -10
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_cgetrs_batched(
        handle.cast(),
        op,
        n,
        nrhs,
        Aarray.cast(),
        lda,
        devIpiv,
        n.into(),
        Barray.cast(),
        ldb,
        batchSize,
    ))
}

unsafe fn strsm_batched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f32,
    A: *const *const f32,
    lda: ::std::os::raw::c_int,
    B: *const *mut f32,
    ldb: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(side);
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_strsm_batched(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha,
        A,
        lda,
        B,
        ldb,
        batchCount,
    ))
}

unsafe fn dtrsm_batched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const f64,
    A: *const *const f64,
    lda: ::std::os::raw::c_int,
    B: *const *mut f64,
    ldb: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(side);
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_dtrsm_batched(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha,
        A,
        lda,
        B,
        ldb,
        batchCount,
    ))
}

unsafe fn ctrsm_batched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuComplex,
    A: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    B: *const *mut cuComplex,
    ldb: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(side);
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ctrsm_batched(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        batchCount,
    ))
}

unsafe fn ztrsm_batched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    alpha: *const cuDoubleComplex,
    A: *const *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    B: *const *mut cuDoubleComplex,
    ldb: ::std::os::raw::c_int,
    batchCount: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(side);
    let fillMode = to_fill(uplo);
    let op = op_from_cuda(trans);
    let diag_type = to_diag(diag);
    to_cuda(rocblas_ztrsm_batched(
        handle.cast(),
        side_mode,
        fillMode,
        op,
        diag_type,
        m,
        n,
        alpha.cast(),
        A.cast(),
        lda,
        B.cast(),
        ldb,
        batchCount,
    ))
}

unsafe fn smatinv_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const f32,
    lda: ::std::os::raw::c_int,
    Ainv: *const *mut f32,
    lda_inv: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn dmatinv_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const f64,
    lda: ::std::os::raw::c_int,
    Ainv: *const *mut f64,
    lda_inv: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn cmatinv_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const cuComplex,
    lda: ::std::os::raw::c_int,
    Ainv: *const *mut cuComplex,
    lda_inv: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn zmatinv_batched(
    handle: cublasHandle_t,
    n: ::std::os::raw::c_int,
    A: *const *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Ainv: *const *mut cuDoubleComplex,
    lda_inv: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn sgeqrf_batched(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Aarray: *const *mut f32,
    lda: ::std::os::raw::c_int,
    TauArray: *const *mut f32,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if m > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if m < 0 {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -3;
    } else if lda < max {
        *info = -4;
    } else if TauArray.is_null() && m * n != 0 {
        *info = -5;
    } else if batchSize < 0 {
        *info = -7;
    } else {
        *info = 0;
    }
    rocsolver_to_cuda(rocsolver_sys::rocsolver_sgeqrf_ptr_batched(
        handle.cast(),
        m,
        n,
        Aarray,
        lda,
        TauArray as _,
        batchSize,
    ))
}

unsafe fn dgeqrf_batched(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Aarray: *const *mut f64,
    lda: ::std::os::raw::c_int,
    TauArray: *const *mut f64,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if m > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if m < 0 {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -3;
    } else if lda < max {
        *info = -4;
    } else if TauArray.is_null() && m * n != 0 {
        *info = -5;
    } else if batchSize < 0 {
        *info = -7;
    } else {
        *info = 0;
    }
    rocsolver_to_cuda(rocsolver_sys::rocsolver_dgeqrf_ptr_batched(
        handle.cast(),
        m,
        n,
        Aarray,
        lda,
        TauArray as _,
        batchSize,
    ))
}

unsafe fn cgeqrf_batched(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Aarray: *const *mut cuComplex,
    lda: ::std::os::raw::c_int,
    TauArray: *const *mut cuComplex,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if m > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if m < 0 {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -3;
    } else if lda < max {
        *info = -4;
    } else if TauArray.is_null() && m * n != 0 {
        *info = -5;
    } else if batchSize < 0 {
        *info = -7;
    } else {
        *info = 0;
    }
    rocsolver_to_cuda(rocsolver_sys::rocsolver_cgeqrf_ptr_batched(
        handle.cast(),
        m,
        n,
        Aarray.cast(),
        lda,
        TauArray as _,
        batchSize,
    ))
}

unsafe fn zgeqrf_batched(
    handle: cublasHandle_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    Aarray: *const *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    TauArray: *const *mut cuDoubleComplex,
    info: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let max = if m > 1 { n } else { 1 };
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if m < 0 {
        *info = -1;
    } else if n < 0 {
        *info = -2;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -3;
    } else if lda < max {
        *info = -4;
    } else if TauArray.is_null() && m * n != 0 {
        *info = -5;
    } else if batchSize < 0 {
        *info = -7;
    } else {
        *info = 0;
    }
    rocsolver_to_cuda(rocsolver_sys::rocsolver_cgeqrf_ptr_batched(
        handle.cast(),
        m,
        n,
        Aarray.cast(),
        lda,
        TauArray as _,
        batchSize,
    ))
}

unsafe fn sgels_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *mut f32,
    lda: ::std::os::raw::c_int,
    Carray: *const *mut f32,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    devInfoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N && trans != cublasOperation_t::CUBLAS_OP_T {
        *info = -1;
    } else if m < 0 {
        *info = -2;
    } else if n < 0 {
        *info = -3;
    } else if nrhs < 0 {
        *info = -4;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -5;
    } else if lda < m {
        *info = -6;
    } else if Carray.is_null() && (m * nrhs != 0 || n * nrhs != 0) {
        *info = -7;
    } else if ldc < m || ldc < n {
        *info = -8;
    } else if devInfoArray.is_null() && batchSize != 0 {
        *info = -10;
    } else if batchSize < 0 {
        *info = -11;
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_sgels_batched(
        handle.cast(),
        op,
        m,
        n,
        nrhs,
        Aarray,
        lda,
        Carray,
        ldc,
        devInfoArray,
        batchSize,
    ))
}

unsafe fn dgels_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *mut f64,
    lda: ::std::os::raw::c_int,
    Carray: *const *mut f64,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    devInfoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N && trans != cublasOperation_t::CUBLAS_OP_T {
        *info = -1;
    } else if m < 0 {
        *info = -2;
    } else if n < 0 {
        *info = -3;
    } else if nrhs < 0 {
        *info = -4;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -5;
    } else if lda < m {
        *info = -6;
    } else if Carray.is_null() && (m * nrhs != 0 || n * nrhs != 0) {
        *info = -7;
    } else if ldc < m || ldc < n {
        *info = -8;
    } else if devInfoArray.is_null() && batchSize != 0 {
        *info = -10;
    } else if batchSize < 0 {
        *info = -11;
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_dgels_batched(
        handle.cast(),
        op,
        m,
        n,
        nrhs,
        Aarray,
        lda,
        Carray,
        ldc,
        devInfoArray,
        batchSize,
    ))
}

unsafe fn cgels_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *mut cuComplex,
    lda: ::std::os::raw::c_int,
    Carray: *const *mut cuComplex,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    devInfoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N && trans != cublasOperation_t::CUBLAS_OP_T {
        *info = -1;
    } else if m < 0 {
        *info = -2;
    } else if n < 0 {
        *info = -3;
    } else if nrhs < 0 {
        *info = -4;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -5;
    } else if lda < m {
        *info = -6;
    } else if Carray.is_null() && (m * nrhs != 0 || n * nrhs != 0) {
        *info = -7;
    } else if ldc < m || ldc < n {
        *info = -8;
    } else if devInfoArray.is_null() && batchSize != 0 {
        *info = -10;
    } else if batchSize < 0 {
        *info = -11;
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_cgels_batched(
        handle.cast(),
        op,
        m,
        n,
        nrhs,
        Aarray.cast(),
        lda,
        Carray.cast(),
        ldc,
        devInfoArray,
        batchSize,
    ))
}

unsafe fn zgels_batched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    nrhs: ::std::os::raw::c_int,
    Aarray: *const *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    Carray: *const *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
    info: *mut ::std::os::raw::c_int,
    devInfoArray: *mut ::std::os::raw::c_int,
    batchSize: ::std::os::raw::c_int,
) -> cublasStatus_t {
    if info.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    } else if trans != cublasOperation_t::CUBLAS_OP_N && trans != cublasOperation_t::CUBLAS_OP_T {
        *info = -1;
    } else if m < 0 {
        *info = -2;
    } else if n < 0 {
        *info = -3;
    } else if nrhs < 0 {
        *info = -4;
    } else if Aarray.is_null() && m * n != 0 {
        *info = -5;
    } else if lda < m {
        *info = -6;
    } else if Carray.is_null() && (m * nrhs != 0 || n * nrhs != 0) {
        *info = -7;
    } else if ldc < m || ldc < n {
        *info = -8;
    } else if devInfoArray.is_null() && batchSize != 0 {
        *info = -10;
    } else if batchSize < 0 {
        *info = -11;
    } else {
        *info = 0;
    }
    let op = rocsolver_op_from_cuda(trans);
    rocsolver_to_cuda(rocsolver_sys::rocsolver_zgels_batched(
        handle.cast(),
        op,
        m,
        n,
        nrhs,
        Aarray.cast(),
        lda,
        Carray.cast(),
        ldc,
        devInfoArray,
        batchSize,
    ))
}

unsafe fn sdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    x: *const f32,
    incx: ::std::os::raw::c_int,
    C: *mut f32,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(mode);
    to_cuda(rocblas_sdgmm(
        handle.cast(),
        side_mode,
        m,
        n,
        A,
        lda,
        x,
        incx,
        C,
        ldc,
    ))
}

unsafe fn ddgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    x: *const f64,
    incx: ::std::os::raw::c_int,
    C: *mut f64,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(mode);
    to_cuda(rocblas_ddgmm(
        handle.cast(),
        side_mode,
        m,
        n,
        A,
        lda,
        x,
        incx,
        C,
        ldc,
    ))
}

unsafe fn cdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuComplex,
    incx: ::std::os::raw::c_int,
    C: *mut cuComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(mode);
    to_cuda(rocblas_cdgmm(
        handle.cast(),
        side_mode,
        m,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
        C.cast(),
        ldc,
    ))
}

unsafe fn zdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    x: *const cuDoubleComplex,
    incx: ::std::os::raw::c_int,
    C: *mut cuDoubleComplex,
    ldc: ::std::os::raw::c_int,
) -> cublasStatus_t {
    let side_mode = to_side(mode);
    to_cuda(rocblas_cdgmm(
        handle.cast(),
        side_mode,
        m,
        n,
        A.cast(),
        lda,
        x.cast(),
        incx,
        C.cast(),
        ldc,
    ))
}

unsafe fn stpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    AP: *const f32,
    A: *mut f32,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn dtpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    AP: *const f64,
    A: *mut f64,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ctpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    AP: *const cuComplex,
    A: *mut cuComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ztpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    AP: *const cuDoubleComplex,
    A: *mut cuDoubleComplex,
    lda: ::std::os::raw::c_int,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn strttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f32,
    lda: ::std::os::raw::c_int,
    AP: *mut f32,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn dtrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const f64,
    lda: ::std::os::raw::c_int,
    AP: *mut f64,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ctrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuComplex,
    lda: ::std::os::raw::c_int,
    AP: *mut cuComplex,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

unsafe fn ztrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: ::std::os::raw::c_int,
    A: *const cuDoubleComplex,
    lda: ::std::os::raw::c_int,
    AP: *mut cuDoubleComplex,
) -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}
