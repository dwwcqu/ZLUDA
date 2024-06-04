#![allow(non_snake_case)]
#[allow(warnings)]
mod cudnn_types_v7;
#[allow(warnings)]
mod cudnn_types_v8;

pub mod types {
    pub use super::cudnn_types_v7::*;
    pub use super::cudnn_types_v8::*;
}

#[allow(warnings)]
mod cudnn_v7;
pub use cudnn_v7::*;

#[allow(warnings)]
mod cudnn_v8;
pub use cudnn_v8::*;

use types::*;

use hip_runtime_sys::*;
use miopen_sys::*;
use std::{mem, ptr};

macro_rules! call {
    ($expr:expr) => {{
        let result = $expr;
        if result != miopen_sys::miopenStatus_t::miopenStatusSuccess {
            return to_cudnn(result);
        }
    }};
}

#[cfg(debug_assertions)]
fn unsupported() -> cudnnStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

fn to_cudnn(status: miopen_sys::miopenStatus_t) -> cudnnStatus_t {
    match status {
        miopen_sys::miopenStatus_t::miopenStatusSuccess => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        miopenStatus_t::miopenStatusNotInitialized => cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
        miopenStatus_t::miopenStatusInvalidValue => cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE,
        miopenStatus_t::miopenStatusBadParm => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
        miopenStatus_t::miopenStatusAllocFailed => cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED,
        miopenStatus_t::miopenStatusInternalError => cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
        miopenStatus_t::miopenStatusVersionMismatch => cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH,
        err => panic!("{}", err.0), //cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
    }
}

unsafe fn create(handle: *mut cudnnHandle_t) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreate(handle as _))
}

unsafe fn cudnn_create_tensor_descriptor(
    tensor_desc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreateTensorDescriptor(tensor_desc as _))
}

unsafe fn cudnn_create_activation_descriptor(
    activation_desc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreateActivationDescriptor(
        activation_desc as _,
    ))
}

unsafe fn cudnn_create_convolution_descriptor(
    conv_desc: *mut cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreateConvolutionDescriptor(
        conv_desc as _,
    ))
}

unsafe fn cudnn_create_filter_descriptor(
    filter_desc: *mut cudnnFilterDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreateTensorDescriptor(filter_desc as _))
}

unsafe fn cudnn_create_lrn_descriptor(norm_desc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreateLRNDescriptor(norm_desc as _))
}

unsafe fn cudnn_create_pooling_descriptor(
    pooling_desc: *mut cudnnPoolingDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenCreatePoolingDescriptor(pooling_desc as _))
}

unsafe fn set_tensor_nd_decriptor(
    tensor_desc: *mut cudnnTensorStruct,
    data_type: cudnnDataType_t,
    nb_dims: i32,
    dim_a: *const i32,
    stride_a: *const i32,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenSetTensorDescriptor(
        tensor_desc as _,
        from_data_type(data_type),
        nb_dims,
        dim_a as _,
        stride_a as _,
    ))
}

fn from_data_type(type_: cudnnDataType_t) -> miopenDataType_t {
    match type_ {
        cudnnDataType_t::CUDNN_DATA_FLOAT => miopenDataType_t::miopenFloat,
        cudnnDataType_t::CUDNN_DATA_DOUBLE => miopenDataType_t::miopenDouble,
        cudnnDataType_t::CUDNN_DATA_HALF => miopenDataType_t::miopenHalf,
        cudnnDataType_t::CUDNN_DATA_INT32 => miopenDataType_t::miopenInt32,
        cudnnDataType_t::CUDNN_DATA_INT8 => miopenDataType_t::miopenInt8,
        cudnnDataType_t::CUDNN_DATA_INT8x4 => miopenDataType_t::miopenInt8x4,
        cudnnDataType_t::CUDNN_DATA_BFLOAT16 => miopenDataType_t::miopenBFloat16,
        _ => todo!(),
    }
}

fn miopen_datatype_to_cudnn(type_: miopenDataType_t) -> cudnnDataType_t {
    match type_ {
        miopenDataType_t::miopenFloat => cudnnDataType_t::CUDNN_DATA_FLOAT,
        miopenDataType_t::miopenDouble => cudnnDataType_t::CUDNN_DATA_DOUBLE,
        miopenDataType_t::miopenHalf => cudnnDataType_t::CUDNN_DATA_HALF,
        miopenDataType_t::miopenInt32 => cudnnDataType_t::CUDNN_DATA_INT32,
        miopenDataType_t::miopenInt8 => cudnnDataType_t::CUDNN_DATA_INT8,
        miopenDataType_t::miopenInt8x4 => cudnnDataType_t::CUDNN_DATA_INT8x4,
        miopenDataType_t::miopenBFloat16 => cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        _ => todo!(),
    }
}

fn to_batch_mode(mode: cudnnBatchNormMode_t) -> miopenBatchNormMode_t {
    match mode {
        cudnnBatchNormMode_t::CUDNN_BATCHNORM_PER_ACTIVATION => {
            miopenBatchNormMode_t::miopenBNPerActivation
        }
        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL => miopenBatchNormMode_t::miopenBNSpatial,
        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT => {
            miopenBatchNormMode_t::miopenBNSpatial
        }
        _ => todo!(),
    }
}

fn to_pooling_mode(mode: cudnnPoolingMode_t) -> miopenPoolingMode_t {
    match mode {
        cudnnPoolingMode_t::CUDNN_POOLING_MAX_DETERMINISTIC => {
            miopenPoolingMode_t::miopenPoolingMax
        }
        cudnnPoolingMode_t::CUDNN_POOLING_MAX => miopenPoolingMode_t::miopenPoolingMax,
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverageInclusive
        }
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverage
        }
        _ => todo!(),
    }
}

fn miopen_to_pooling_mode(mode: miopenPoolingMode_t) -> cudnnPoolingMode_t {
    match mode {
        miopenPoolingMode_t::miopenPoolingMax => {
            cudnnPoolingMode_t::CUDNN_POOLING_MAX_DETERMINISTIC
        }
        miopenPoolingMode_t::miopenPoolingAverageInclusive => {
            cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        }
        miopenPoolingMode_t::miopenPoolingAverage => {
            cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
        }
        _ => todo!(),
    }
}

unsafe fn set_filter_nd_descriptor(
    filter_desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    _format: cudnnTensorFormat_t,
    nb_dims: i32,
    filter_dim_a: *const i32,
) -> cudnnStatus_t {
    let data_type = from_data_type(data_type);
    to_cudnn(miopenSetTensorDescriptor(
        filter_desc as _,
        data_type,
        nb_dims,
        filter_dim_a as _,
        ptr::null_mut(),
    ))
}

unsafe fn set_convolution_nd_descriptor(
    conv_desc: cudnnConvolutionDescriptor_t,
    array_length: i32,
    pad_a: *const i32,
    filter_stride_a: *const i32,
    dilation_a: *const i32,
    mode: cudnnConvolutionMode_t,
    _compute_type: cudnnDataType_t,
) -> cudnnStatus_t {
    if array_length != 2 {
        todo!()
    }
    let pad_h = *pad_a.add(0);
    let pad_w = *pad_a.add(1);
    let u = *filter_stride_a.add(0);
    let v = *filter_stride_a.add(1);
    let d_h = *dilation_a.add(0);
    let d_w = *dilation_a.add(1);
    let mode = conv_mode_to_cudnn(mode);
    to_cudnn(miopen_sys::miopenInitConvolutionDescriptor(
        conv_desc as _,
        mode,
        pad_h,
        pad_w,
        u,
        v,
        d_h,
        d_w,
    ))
}

fn conv_mode_to_cudnn(mode: cudnnConvolutionMode_t) -> miopenConvolutionMode_t {
    match mode {
        cudnnConvolutionMode_t::CUDNN_CONVOLUTION => miopenConvolutionMode_t::miopenTranspose,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION => {
            miopenConvolutionMode_t::miopenConvolution
        }
        _ => panic!(),
    }
}

unsafe fn get_convolution_nd_forward_output_dim(
    conv_desc: cudnnConvolutionDescriptor_t,
    input_tensor_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    mut nb_dims: i32,
    tensor_ouput_dim_a: *mut i32,
) -> cudnnStatus_t {
    to_cudnn(miopen_sys::miopenGetConvolutionNdForwardOutputDim(
        conv_desc as _,
        input_tensor_desc as _,
        filter_desc as _,
        &mut nb_dims as *mut _,
        tensor_ouput_dim_a,
    ))
}

unsafe fn find_convolution_forward_algorithm(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t {
    let mut result = vec![mem::zeroed(); requested_algo_count as usize];
    let mut x_size = 0;
    call! { miopenGetTensorNumBytes(x_desc as _, &mut x_size) };
    let mut x = mem::zeroed();
    let error = hipMalloc(&mut x, x_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut w_size = 0;
    call! { miopenGetTensorNumBytes(w_desc as _, &mut w_size) };
    let mut w = mem::zeroed();
    let error = hipMalloc(&mut w, w_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut y_size = 0;
    call! { miopenGetTensorNumBytes(y_desc as _, &mut y_size) };
    let mut y = mem::zeroed();
    let error = hipMalloc(&mut y, y_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut workspace_size = 0;
    call! { miopenConvolutionForwardGetWorkSpaceSize(handle as _, w_desc as _, x_desc as _, conv_desc as _, y_desc as _, &mut workspace_size) };
    let mut workspace = mem::zeroed();
    let error = hipMalloc(&mut workspace, workspace_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = to_cudnn(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        requested_algo_count,
        returned_algo_count,
        result.as_mut_ptr(),
        workspace,
        workspace_size,
        true,
    ));
    // TODO: propagaate error codes
    drop(hipFree(x));
    drop(hipFree(w));
    drop(hipFree(y));
    drop(hipFree(workspace));
    for i in 0..result.len() {
        let result = result[i];
        *perf_results.add(i) = algoperf_to_cudnn(result);
    }
    error
}

unsafe fn find_convolution_forward_algorithm_ex(
    handle: *mut cudnnContext,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionFwdAlgoPerfStruct,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    let mut result = vec![mem::zeroed(); requested_algo_count as usize];
    let error = to_cudnn(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        requested_algo_count,
        returned_algo_count,
        result.as_mut_ptr(),
        work_space,
        work_space_size_in_bytes,
        true,
    ));
    for i in 0..result.len() {
        let result = result[i];
        *perf_results.add(i) = algoperf_to_cudnn(result);
    }
    error
}

unsafe fn algoperf_to_cudnn(result: miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgoPerf_t {
    let algo = algo_to_cudnn(result);
    cudnnConvolutionFwdAlgoPerf_t {
        algo,
        status: cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        time: result.time,
        memory: result.memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: mem::zeroed(),
    }
}

unsafe fn algo_to_cudnn(result: miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgo_t {
    match result.__bindgen_anon_1.fwd_algo {
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoDirect => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        }
        _ => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    }
}

unsafe fn get_convolution_forward_algorithm(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    _memory_limit_in_bytes: usize,
    algo: *mut cudnnConvolutionFwdAlgo_t,
) -> cudnnStatus_t {
    let mut algo_count = 0;
    let mut result = mem::zeroed();
    let mut x_size = 0;
    call! { miopenGetTensorNumBytes(x_desc as _, &mut x_size) };
    let mut x = mem::zeroed();
    let error = hipMalloc(&mut x, x_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut w_size = 0;
    call! { miopenGetTensorNumBytes(w_desc as _, &mut w_size) };
    let mut w = mem::zeroed();
    let error = hipMalloc(&mut w, w_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut y_size = 0;
    call! { miopenGetTensorNumBytes(y_desc as _, &mut y_size) };
    let mut y = mem::zeroed();
    let error = hipMalloc(&mut y, y_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut workspace_size = 0;
    call! { miopenConvolutionForwardGetWorkSpaceSize(handle as _, w_desc as _, x_desc as _, conv_desc as _, y_desc as _, &mut workspace_size) };
    let mut workspace = mem::zeroed();
    let error = hipMalloc(&mut workspace, workspace_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = to_cudnn(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        1,
        &mut algo_count,
        &mut result,
        workspace,
        workspace_size,
        true,
    ));
    // TODO: propagate error codes
    drop(hipFree(x));
    drop(hipFree(w));
    drop(hipFree(y));
    drop(hipFree(workspace));
    if algo_count > 0 {
        *algo = algo_to_cudnn(result);
    }
    error
}

pub unsafe fn get_convolution_forward_workspace_size(
    handle: *mut cudnnContext,
    x_desc: *mut cudnnTensorStruct,
    w_desc: *mut cudnnFilterStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    y_desc: *mut cudnnTensorStruct,
    _algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenConvolutionForwardGetWorkSpaceSize(
        handle as _,
        w_desc as _,
        x_desc as _,
        conv_desc as _,
        y_desc as _,
        size_in_bytes,
    ))
}

unsafe fn convolution_forward(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let mut algo = algo_from_cudnn(algo);
    // In cuDNN it is possible to find algorithm for sizes X and then pass the algo
    // for sizes Y. On miOpen this fails
    let mut perf_results = vec![mem::zeroed(); 32];
    let mut algo_count = 0;
    call!(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        32,
        &mut algo_count,
        perf_results.as_mut_ptr(),
        work_space,
        work_space_size_in_bytes,
        true,
    ));
    if algo_count == 0 {
        panic!()
    }
    if let None = perf_results[..algo_count as usize]
        .iter()
        .find(|result| result.__bindgen_anon_1.fwd_algo == algo)
    {
        algo = perf_results[0].__bindgen_anon_1.fwd_algo;
    }
    to_cudnn(miopenConvolutionForward(
        handle as _,
        alpha,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        algo,
        beta,
        y_desc as _,
        y,
        work_space,
        work_space_size_in_bytes,
    ))
}

fn algo_from_cudnn(algo: cudnnConvolutionFwdAlgo_t) -> miopenConvFwdAlgorithm_t {
    match algo {
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoDirect
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd
        }
        _ => panic!(),
    }
}

unsafe fn add_tensor(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    a_desc: *mut cudnnTensorStruct,
    a: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c_desc: *mut cudnnTensorStruct,
    c: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    // CUDA tensor A might be 1 in some dimensions
    // MIOpen tensors A and C must be the same
    let zero = 0f64;
    to_cudnn(miopenOpTensor(
        handle as _,
        miopenTensorOp_t::miopenTensorOpAdd,
        alpha,
        c_desc as _,
        c,
        beta,
        a_desc as _,
        a,
        &zero as *const _ as _,
        c_desc as _,
        c,
    ))
}

unsafe fn set_pooling_nd_descriptor(
    pooling_desc: *mut cudnnPoolingStruct,
    mode: cudnnPoolingMode_t,
    _maxpooling_nan_opt: cudnnNanPropagation_t,
    nb_dims: i32,
    window_dim_a: *const i32,
    padding_a: *const i32,
    stride_a: *const i32,
) -> cudnnStatus_t {
    let mode = pooling_from_cudnn(mode);
    to_cudnn(miopenSetNdPoolingDescriptor(
        pooling_desc as _,
        mode,
        nb_dims,
        window_dim_a as _,
        padding_a as _,
        stride_a as _,
    ))
}

fn pooling_from_cudnn(mode: cudnnPoolingMode_t) -> miopenPoolingMode_t {
    match mode {
        cudnnPoolingMode_t::CUDNN_POOLING_MAX => miopenPoolingMode_t::miopenPoolingMax,
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverageInclusive
        }
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverage
        }
        _ => todo!(),
    }
}

unsafe fn get_pooling_nd_forward_output_dim(
    pooling_desc: *mut cudnnPoolingStruct,
    input_tensor_desc: *mut cudnnTensorStruct,
    nb_dims: i32,
    output_tensor_dim_a: *mut i32,
) -> cudnnStatus_t {
    if nb_dims != 4 {
        todo!()
    }
    to_cudnn(miopenGetPoolingForwardOutputDim(
        pooling_desc as _,
        input_tensor_desc as _,
        output_tensor_dim_a.add(0),
        output_tensor_dim_a.add(1),
        output_tensor_dim_a.add(2),
        output_tensor_dim_a.add(3),
    ))
}

unsafe fn pooling_forward(
    handle: *mut cudnnContext,
    pooling_desc: *mut cudnnPoolingStruct,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let mut workspace_size = 0;
    call! { miopenPoolingGetWorkSpaceSize(y_desc as _, &mut workspace_size) };
    let mut workspace = mem::zeroed();
    let error = hipMalloc(&mut workspace, workspace_size);
    if error != hipError_t::hipSuccess {
        return cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR;
    }
    // TODO: Only alpha=1 and beta=0 is supported
    let error = to_cudnn(miopenPoolingForward(
        handle as _,
        pooling_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        false,
        workspace,
        workspace_size,
    ));
    // TODO: propagate error codes
    drop(hipFree(workspace));
    error
}

unsafe fn set_activation_descriptor(
    activation_desc: *mut cudnnActivationStruct,
    mode: cudnnActivationMode_t,
    _relu_nan_opt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t {
    let mode = activation_mode(mode);
    to_cudnn(miopenSetActivationDescriptor(
        activation_desc as _,
        mode,
        coef,
        0.0,
        0.0,
    ))
}

fn activation_mode(mode: cudnnActivationMode_t) -> miopenActivationMode_t {
    match mode {
        cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID => {
            miopenActivationMode_t::miopenActivationLOGISTIC
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_RELU => {
            miopenActivationMode_t::miopenActivationRELU
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_TANH => {
            miopenActivationMode_t::miopenActivationTANH
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU => {
            miopenActivationMode_t::miopenActivationCLIPPEDRELU
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_ELU => miopenActivationMode_t::miopenActivationELU,
        cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY => {
            miopenActivationMode_t::miopenActivationPASTHRU
        }
        _ => panic!(),
    }
}

unsafe fn activation_forward(
    handle: *mut cudnnContext,
    activation_desc: *mut cudnnActivationStruct,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenActivationForward(
        handle as _,
        activation_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
    ))
}

unsafe fn set_lrn_descriptor(
    norm_desc: *mut cudnnLRNStruct,
    lrn_n: u32,
    lrn_alpha: f64,
    lrn_beta: f64,
    lrn_k: f64,
) -> cudnnStatus_t {
    to_cudnn(miopenSetLRNDescriptor(
        norm_desc as _,
        miopenLRNMode_t::miopenLRNCrossChannel, // ???
        lrn_n,
        lrn_alpha,
        lrn_beta,
        lrn_k,
    ))
}

unsafe fn lrn_cross_channel_forward(
    handle: *mut cudnnContext,
    norm_desc: *mut cudnnLRNStruct,
    _lrn_mode: cudnnLRNMode_t,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenLRNForward(
        handle as _,
        norm_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        false,
        ptr::null_mut(),
    ))
}

unsafe fn softmax_forward(
    handle: *mut cudnnContext,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let algo = softmax_algo(algo);
    let mode = softmax_mode(mode);
    to_cudnn(miopenSoftmaxForward_V2(
        handle as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        algo,
        mode,
    ))
}

fn softmax_algo(algo: cudnnSoftmaxAlgorithm_t) -> miopenSoftmaxAlgorithm_t {
    match algo {
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE => {
            miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_ACCURATE
        }
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST => {
            miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_FAST
        }
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG => miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_LOG,
        _ => panic!(),
    }
}

fn softmax_mode(mode: cudnnSoftmaxMode_t) -> miopenSoftmaxMode_t {
    match mode {
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL => {
            miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_CHANNEL
        }
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE => {
            miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_INSTANCE
        }
        _ => panic!(),
    }
}

unsafe fn destroy(handle: *mut cudnnContext) -> cudnnStatus_t {
    to_cudnn(miopenDestroy(handle as _))
}

unsafe fn destroy_activation_descriptor(
    activation_desc: *mut cudnnActivationStruct,
) -> cudnnStatus_t {
    to_cudnn(miopenDestroyActivationDescriptor(activation_desc as _))
}

unsafe fn destroy_convolution_descriptor(conv_desc: *mut cudnnConvolutionStruct) -> cudnnStatus_t {
    to_cudnn(miopenDestroyConvolutionDescriptor(conv_desc as _))
}

unsafe fn destroy_filter_descriptor(filter_desc: *mut cudnnFilterStruct) -> cudnnStatus_t {
    to_cudnn(miopenDestroyTensorDescriptor(filter_desc as _))
}

unsafe fn destroy_lrn_descriptor(lrn_desc: *mut cudnnLRNStruct) -> cudnnStatus_t {
    to_cudnn(miopenDestroyLRNDescriptor(lrn_desc as _))
}

unsafe fn destroy_pooling_descriptor(pooling_desc: *mut cudnnPoolingStruct) -> cudnnStatus_t {
    to_cudnn(miopenDestroyPoolingDescriptor(pooling_desc as _))
}

unsafe fn destroy_tensor_descriptor(tensor_desc: *mut cudnnTensorStruct) -> cudnnStatus_t {
    to_cudnn(miopenDestroyTensorDescriptor(tensor_desc as _))
}

unsafe fn set_tensor_4d_descriptor_ex(
    tensor_desc: *mut cudnnTensorStruct,
    data_type: cudnnDataType_t,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    n_stride: i32,
    c_stride: i32,
    h_stride: i32,
    w_stride: i32,
) -> cudnnStatus_t {
    let data_type = from_data_type(data_type);
    to_cudnn(miopenSet4dTensorDescriptorEx(
        tensor_desc as _,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    ))
}

unsafe fn transform_tensor(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenTransformTensor(
        handle as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
    ))
}

unsafe fn set_stream(stream_id: *mut CUstream_st) -> cudnnStatus_t {
    if stream_id != ptr::null_mut() {
        todo!()
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

fn set_convolution_math_type(
    _conv_desc: cudnnConvolutionDescriptor_t,
    _math_type: cudnnMathType_t,
) -> cudnnStatus_t {
    //TODO: implement
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn set_convolution_group_count(
    conv_desc: *mut cudnnConvolutionStruct,
    group_count: i32,
) -> cudnnStatus_t {
    //TODO: implement
    to_cudnn(miopenSetConvolutionGroupCount(conv_desc as _, group_count))
}

unsafe fn get_convolution_backward_data_algorithm_max_count(
    _handle: *mut cudnnContext,
    count: *mut i32,
) -> cudnnStatus_t {
    *count = 1;
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn get_convolution_backward_data_algorithm_v7(
    handle: *mut cudnnContext,
    w_desc: *mut cudnnFilterStruct,
    dy_desc: *mut cudnnTensorStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    dx_desc: *mut cudnnTensorStruct,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionBwdDataAlgoPerf_t,
    memory_limit_in_bytes: usize,
) -> cudnnStatus_t {
    let mut work_space_size = 0;
    let mut dy_size = 0;
    call! { miopenGetTensorNumBytes(dy_desc as _, &mut dy_size) };
    let mut dy = mem::zeroed();
    let error = hipMalloc(&mut dy, dy_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut w_size = 0;
    call! { miopenGetTensorNumBytes(w_desc as _, &mut w_size) };
    let mut w = mem::zeroed();
    let error = hipMalloc(&mut w, w_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut dx_size = 0;
    call! { miopenGetTensorNumBytes(dx_desc as _, &mut dx_size) };
    let mut dx = mem::zeroed();
    let error = hipMalloc(&mut dx, dx_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = to_cudnn(miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle as _,
        dy_desc as _,
        w_desc as _,
        conv_desc as _,
        dx_desc as _,
        &mut work_space_size,
    ));
    work_space_size = work_space_size.min(memory_limit_in_bytes);
    if error != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        panic!("")
    }
    let mut work_space = mem::zeroed();
    if hipMalloc(&mut work_space, work_space_size) != hipError_t::hipSuccess {
        panic!("")
    }
    let mut miopen_perf_results = vec![mem::zeroed(); requested_algo_count as usize];
    let result = to_cudnn(miopenFindConvolutionBackwardDataAlgorithm(
        handle as _,
        dy_desc as _,
        dy,
        w_desc as _,
        w,
        conv_desc as _,
        dx_desc as _,
        dx,
        requested_algo_count,
        returned_algo_count,
        miopen_perf_results.as_mut_ptr(),
        work_space,
        work_space_size,
        true,
    ));
    drop(hipFree(dy));
    drop(hipFree(w));
    drop(hipFree(dx));
    drop(hipFree(work_space));
    for i in 0..*returned_algo_count {
        *perf_results.add(i as usize) = convert_bwd_algo(miopen_perf_results[i as usize]);
    }
    result
}

unsafe fn convert_bwd_algo(result: miopenConvAlgoPerf_t) -> cudnnConvolutionBwdDataAlgoPerf_t {
    let algo = bwd_data_algo_to_cudnn(result.__bindgen_anon_1.bwd_data_algo);
    cudnnConvolutionBwdDataAlgoPerf_t {
        algo,
        status: cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        time: result.time,
        memory: result.memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: mem::zeroed(),
    }
}

fn bwd_data_algo_to_cudnn(algo: miopenConvBwdDataAlgorithm_t) -> cudnnConvolutionBwdDataAlgo_t {
    match algo {
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoDirect => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
        }
        miopenConvBwdDataAlgorithm_t::miopenTransposeBwdDataAlgoGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoImplicitGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        _ => panic!(),
    }
}

fn bwd_data_algo_from_cudnn(algo: cudnnConvolutionBwdDataAlgo_t) -> miopenConvBwdDataAlgorithm_t {
    match algo {
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoGEMM
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoDirect
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd
        }
        _ => panic!(),
    }
}

unsafe fn get_convolution_backward_data_algorithm(
    handle: *mut cudnnContext,
    w_desc: *mut cudnnFilterStruct,
    dy_desc: *mut cudnnTensorStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    dx_desc: *mut cudnnTensorStruct,
    memory_limit_in_bytes: usize,
    algo: *mut cudnnConvolutionBwdDataAlgo_t,
) -> cudnnStatus_t {
    let mut algo_count = 0;
    let mut perf_result = mem::zeroed::<cudnnConvolutionBwdDataAlgoPerf_t>();
    let error = get_convolution_backward_data_algorithm_v7(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        1,
        &mut algo_count,
        &mut perf_result as *mut _,
        memory_limit_in_bytes,
    );
    if error != cudnnStatus_t::CUDNN_STATUS_SUCCESS || algo_count == 0 {
        panic!("")
    }
    *algo = perf_result.algo;
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn get_convolution_backward_data_workspace_size(
    handle: *mut cudnnContext,
    w_desc: *mut cudnnFilterStruct,
    dy_desc: *mut cudnnTensorStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    dx_desc: *mut cudnnTensorStruct,
    _algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle as _,
        dy_desc as _,
        w_desc as _,
        conv_desc as _,
        dx_desc as _,
        size_in_bytes,
    ))
}

unsafe fn convolution_backward_data(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    dy_desc: *mut cudnnTensorStruct,
    dy: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    algo: cudnnConvolutionBwdDataAlgo_t,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const std::ffi::c_void,
    dx_desc: *mut cudnnTensorStruct,
    dx: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let algo = bwd_data_algo_from_cudnn(algo);
    to_cudnn(miopenConvolutionBackwardData(
        handle as _,
        alpha,
        dy_desc as _,
        dy,
        w_desc as _,
        w,
        conv_desc as _,
        algo,
        beta,
        dx_desc as _,
        dx,
        work_space,
        work_space_size_in_bytes,
    ))
}

unsafe fn get_convolution_backward_filter_algorithm(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    preference: cudnnConvolutionBwdFilterPreference_t,
    memoryLimitInBytes: usize,
    algo: *mut cudnnConvolutionBwdFilterAlgo_t,
) -> cudnnStatus_t {
    // ! To Do: hipdnnGetConvolutionBackwardFilterAlgorithm() in hipDNN impl
    crate::unsupported()
}

unsafe fn get_stream(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t {
    to_cudnn(miopenGetStream(handle.cast(), streamId.cast()))
}

unsafe fn set_tensor4d_descriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: ::std::os::raw::c_int,
    c: ::std::os::raw::c_int,
    h: ::std::os::raw::c_int,
    w: ::std::os::raw::c_int,
) -> cudnnStatus_t {
    let data_type = from_data_type(dataType);
    to_cudnn(miopenSet4dTensorDescriptor(
        tensorDesc.cast(),
        data_type,
        n,
        c,
        h,
        w,
    ))
}

unsafe fn get_tensor4d_descriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: *mut cudnnDataType_t,
    n: *mut ::std::os::raw::c_int,
    c: *mut ::std::os::raw::c_int,
    h: *mut ::std::os::raw::c_int,
    w: *mut ::std::os::raw::c_int,
    nStride: *mut ::std::os::raw::c_int,
    cStride: *mut ::std::os::raw::c_int,
    hStride: *mut ::std::os::raw::c_int,
    wStride: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    let mut data_type: miopenDataType_t = miopenDataType_t::miopenFloat;
    let status = miopenGet4dTensorDescriptor(
        tensorDesc.cast(),
        &mut data_type,
        n,
        c,
        h,
        w,
        nStride,
        cStride,
        hStride,
        wStride,
    );
    *dataType = miopen_datatype_to_cudnn(data_type);
    to_cudnn(status)
}

unsafe fn get_tensor_nd_descriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    nbDimsRequested: ::std::os::raw::c_int,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut ::std::os::raw::c_int,
    dimA: *mut ::std::os::raw::c_int,
    strideA: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    /*
     * !To Do: hipdnnGetTensorNdDescriptor
     */
    crate::unsupported()
}

unsafe fn create_op_tensor_descriptor(
    opTensorDesc: *mut cudnnOpTensorDescriptor_t,
) -> cudnnStatus_t {
    /*
     * !To Do: hipdnnCreateOpTensorDescriptor
     */
    crate::unsupported()
}

unsafe fn set_op_tensor_descriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: cudnnOpTensorOp_t,
    opTensorCompType: cudnnDataType_t,
    opTensorNanOpt: cudnnNanPropagation_t,
) -> cudnnStatus_t {
    /*
     * !To Do: hipdnnSetOpTensorDescriptor
     */
    crate::unsupported()
}

unsafe fn get_op_tensor_descriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: *mut cudnnOpTensorOp_t,
    opTensorCompType: *mut cudnnDataType_t,
    opTensorNanOpt: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t {
    /*
     * !To Do: hipdnnGetOpTensorDescriptor
     */
    crate::unsupported()
}

unsafe fn destroy_op_tensor_descriptor(opTensorDesc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t {
    /*
     * !To Do: hipdnnDestroyOpTensorDescriptor
     */
    crate::unsupported()
}

unsafe fn convolution_bias_activation_forward(
    handle: cudnnHandle_t,
    alpha1: *const ::std::os::raw::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: *mut ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
    alpha2: *const ::std::os::raw::c_void,
    zDesc: cudnnTensorDescriptor_t,
    z: *const ::std::os::raw::c_void,
    biasDesc: cudnnTensorDescriptor_t,
    bias: *const ::std::os::raw::c_void,
    activationDesc: cudnnActivationDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    let fwd_algo = algo_from_cudnn(algo);
    crate::to_cudnn(miopenConvolutionBiasActivationForward(
        handle.cast(),
        alpha1,
        xDesc.cast(),
        x,
        wDesc.cast(),
        w,
        convDesc.cast(),
        fwd_algo,
        workSpace,
        workSpaceSizeInBytes,
        alpha2,
        zDesc.cast(),
        z,
        biasDesc.cast(),
        bias,
        activationDesc.cast(),
        yDesc.cast(),
        y,
    ))
}

unsafe fn get_convolution2d_forward_output_dim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    n: *mut ::std::os::raw::c_int,
    c: *mut ::std::os::raw::c_int,
    h: *mut ::std::os::raw::c_int,
    w: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    to_cudnn(miopenGetConvolutionForwardOutputDim(
        convDesc.cast(),
        inputTensorDesc.cast(),
        filterDesc.cast(),
        n,
        c,
        h,
        w,
    ))
}

unsafe fn get_ctc_loss_workspace_size(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    gradientsDesc: cudnnTensorDescriptor_t,
    labels: *const ::std::os::raw::c_int,
    labelLengths: *const ::std::os::raw::c_int,
    inputLengths: *const ::std::os::raw::c_int,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenGetCTCLossWorkspaceSize(
        handle.cast(),
        probsDesc.cast(),
        gradientsDesc.cast(),
        labels,
        labelLengths,
        inputLengths,
        miopenCTCLossAlgo_t::MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC,
        ctcLossDesc.cast(),
        sizeInBytes,
    ))
}

unsafe fn ctc_loss(
    handle: cudnnHandle_t,
    probsDesc: cudnnTensorDescriptor_t,
    probs: *const ::std::os::raw::c_void,
    hostLabels: *const ::std::os::raw::c_int,
    hostLabelLengths: *const ::std::os::raw::c_int,
    hostInputLengths: *const ::std::os::raw::c_int,
    costs: *mut ::std::os::raw::c_void,
    gradientsDesc: cudnnTensorDescriptor_t,
    gradients: *mut ::std::os::raw::c_void,
    algo: cudnnCTCLossAlgo_t,
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    workspace: *mut ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenCTCLoss(
        handle.cast(),
        probsDesc.cast(),
        probs,
        hostLabels,
        hostLabelLengths,
        hostInputLengths,
        costs,
        gradientsDesc.cast(),
        gradients,
        miopenCTCLossAlgo_t::MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC,
        ctcLossDesc.cast(),
        workspace,
        workSpaceSizeInBytes,
    ))
}

unsafe fn destroy_ctc_loss_descriptor(ctcLossDesc: cudnnCTCLossDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenDestroyCTCLossDescriptor(ctcLossDesc.cast()))
}

unsafe fn get_ctc_loss_descriptor(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: *mut cudnnDataType_t,
) -> cudnnStatus_t {
    let mut data_type: miopenDataType_t = miopenDataType_t::miopenFloat;
    let mut blank_label_id: i32 = 0;
    let mut apply_softmax_layer: bool = false;
    let status = miopenGetCTCLossDescriptor(
        ctcLossDesc.cast(),
        &mut data_type,
        &mut blank_label_id,
        &mut apply_softmax_layer,
    );
    *compType = miopen_datatype_to_cudnn(data_type);
    to_cudnn(status)
}

unsafe fn set_ctc_loss_descriptor(
    ctcLossDesc: cudnnCTCLossDescriptor_t,
    compType: cudnnDataType_t,
) -> cudnnStatus_t {
    let data_type = from_data_type(compType);
    let blank_label_id: i32 = 0;
    let apply_softmax_layer: bool = false;
    to_cudnn(miopenSetCTCLossDescriptor(
        ctcLossDesc.cast(),
        data_type,
        blank_label_id,
        apply_softmax_layer,
    ))
}

unsafe fn create_ctc_loss_descriptor(ctcLossDesc: *mut cudnnCTCLossDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenCreateCTCLossDescriptor(ctcLossDesc.cast()))
}

unsafe fn rnn_backward_weights(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    xDesc: *const cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    hxDesc: cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    yDesc: *const cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    workSpace: *const ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
    dwDesc: cudnnFilterDescriptor_t,
    dw: *mut ::std::os::raw::c_void,
    reserveSpace: *const ::std::os::raw::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenRNNBackwardWeights(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        xDesc.cast(),
        x,
        hxDesc.cast(),
        hx,
        yDesc.cast(),
        y,
        dwDesc.cast(),
        dw,
        workSpace as _,
        workSpaceSizeInBytes,
        reserveSpace,
        reserveSpaceSizeInBytes,
    ))
}

pub unsafe extern "system" fn rnn_backward_data(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    yDesc: *const cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dyDesc: *const cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dhyDesc: cudnnTensorDescriptor_t,
    dhy: *const ::std::os::raw::c_void,
    dcyDesc: cudnnTensorDescriptor_t,
    dcy: *const ::std::os::raw::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    hxDesc: cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    cxDesc: cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    dxDesc: *const cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    dhxDesc: cudnnTensorDescriptor_t,
    dhx: *mut ::std::os::raw::c_void,
    dcxDesc: cudnnTensorDescriptor_t,
    dcx: *mut ::std::os::raw::c_void,
    workSpace: *mut ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::std::os::raw::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenRNNBackwardData(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        yDesc.cast(),
        y,
        dyDesc.cast(),
        dy,
        dhyDesc.cast(),
        dhy,
        dcyDesc.cast(),
        dcy,
        wDesc.cast(),
        w,
        hxDesc.cast(),
        hx,
        cxDesc.cast(),
        cx,
        dxDesc.cast(),
        dx,
        dhxDesc.cast(),
        dhx,
        dcxDesc.cast(),
        dcx,
        workSpace,
        workSpaceSizeInBytes,
        reserveSpace,
        reserveSpaceSizeInBytes,
    ))
}

unsafe fn rnn_forward_training(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    xDesc: *const cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    hxDesc: cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    cxDesc: cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    yDesc: *const cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    hyDesc: cudnnTensorDescriptor_t,
    hy: *mut ::std::os::raw::c_void,
    cyDesc: cudnnTensorDescriptor_t,
    cy: *mut ::std::os::raw::c_void,
    workSpace: *mut ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
    reserveSpace: *mut ::std::os::raw::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenRNNForwardTraining(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        xDesc.cast(),
        x,
        hxDesc.cast(),
        hx,
        cxDesc.cast(),
        cx,
        wDesc.cast(),
        w,
        yDesc.cast(),
        y,
        hyDesc.cast(),
        hy,
        cyDesc.cast(),
        cy,
        workSpace,
        workSpaceSizeInBytes,
        reserveSpace,
        reserveSpaceSizeInBytes,
    ))
}

unsafe fn rnn_forward_inference(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    xDesc: *const cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    hxDesc: cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    cxDesc: cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    wDesc: cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    yDesc: *const cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    hyDesc: cudnnTensorDescriptor_t,
    hy: *mut ::std::os::raw::c_void,
    cyDesc: cudnnTensorDescriptor_t,
    cy: *mut ::std::os::raw::c_void,
    workSpace: *mut ::std::os::raw::c_void,
    workSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenRNNForwardInference(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        xDesc.cast(),
        x,
        hxDesc.cast(),
        hx,
        cxDesc.cast(),
        cx,
        wDesc.cast(),
        w,
        yDesc.cast(),
        y,
        hyDesc.cast(),
        hy,
        cyDesc.cast(),
        cy,
        workSpace,
        workSpaceSizeInBytes,
    ))
}

unsafe fn get_rnn_params_size(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
    dataType: cudnnDataType_t,
) -> cudnnStatus_t {
    let data_type = from_data_type(dataType);
    to_cudnn(miopenGetRNNParamsSize(
        handle.cast(),
        rnnDesc.cast(),
        xDesc.cast(),
        sizeInBytes,
        data_type,
    ))
}

unsafe fn get_rnn_training_reserve_size(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    xDesc: *const cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenGetRNNTrainingReserveSize(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        xDesc.cast(),
        sizeInBytes,
    ))
}

unsafe fn get_rnn_workspace_size(
    handle: cudnnHandle_t,
    rnnDesc: cudnnRNNDescriptor_t,
    seqLength: ::std::os::raw::c_int,
    xDesc: *const cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenGetRNNWorkspaceSize(
        handle.cast(),
        rnnDesc.cast(),
        seqLength,
        xDesc.cast(),
        sizeInBytes,
    ))
}

unsafe fn destroy_rnn_descriptor(rnnDesc: cudnnRNNDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenDestroyRNNDescriptor(rnnDesc.cast()))
}

unsafe fn create_rnn_descriptor(rnnDesc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenCreateRNNDescriptor(rnnDesc.cast()))
}

unsafe fn dropout_backward(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    dydesc: cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dxdesc: cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    reserveSpace: *mut ::std::os::raw::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenDropoutBackward(
        handle.cast(),
        dropoutDesc.cast(),
        ptr::null_mut(),
        dydesc.cast(),
        dy,
        dxdesc.cast(),
        dx,
        reserveSpace,
        reserveSpaceSizeInBytes,
    ))
}

unsafe fn batch_normalization_backward(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alphaDataDiff: *const ::std::os::raw::c_void,
    betaDataDiff: *const ::std::os::raw::c_void,
    alphaParamDiff: *const ::std::os::raw::c_void,
    betaParamDiff: *const ::std::os::raw::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::std::os::raw::c_void,
    dBnScaleResult: *mut ::std::os::raw::c_void,
    dBnBiasResult: *mut ::std::os::raw::c_void,
    epsilon: f64,
    savedMean: *const ::std::os::raw::c_void,
    savedInvVariance: *const ::std::os::raw::c_void,
) -> cudnnStatus_t {
    let bat_mode = to_batch_mode(mode);
    to_cudnn(miopenBatchNormalizationBackward(
        handle.cast(),
        bat_mode,
        alphaDataDiff,
        betaDataDiff,
        alphaParamDiff,
        betaParamDiff,
        xDesc.cast(),
        x,
        dyDesc.cast(),
        dy,
        dxDesc.cast(),
        dx,
        dBnScaleBiasDesc.cast(),
        bnScale,
        dBnScaleResult,
        dBnBiasResult,
        epsilon,
        savedMean,
        savedInvVariance,
    ))
}

unsafe fn batch_normalization_forward_training(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::std::os::raw::c_void,
    bnBias: *const ::std::os::raw::c_void,
    exponentialAverageFactor: f64,
    resultRunningMean: *mut ::std::os::raw::c_void,
    resultRunningVariance: *mut ::std::os::raw::c_void,
    epsilon: f64,
    resultSaveMean: *mut ::std::os::raw::c_void,
    resultSaveInvVariance: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    let bat_mode = to_batch_mode(mode);
    to_cudnn(miopenBatchNormalizationForwardTraining(
        handle.cast(),
        bat_mode,
        alpha as _,
        beta as _,
        xDesc.cast(),
        x,
        yDesc.cast(),
        y,
        bnScaleBiasMeanVarDesc.cast(),
        bnScale as _,
        bnBias as _,
        exponentialAverageFactor,
        resultRunningMean,
        resultRunningVariance,
        epsilon,
        resultSaveMean,
        resultSaveInvVariance,
    ))
}

unsafe fn activation_backward(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const ::std::os::raw::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenActivationBackward(
        handle.cast(),
        activationDesc.cast(),
        alpha,
        yDesc.cast(),
        y,
        dyDesc.cast(),
        dy,
        xDesc.cast(),
        x,
        beta,
        dxDesc.cast(),
        dx,
    ))
}

unsafe fn softmax_backward(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const ::std::os::raw::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dyDesc: cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    dxDesc: cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenSoftmaxBackward(
        handle.cast(),
        alpha,
        yDesc.cast(),
        y,
        dyDesc.cast(),
        dy,
        beta,
        dxDesc.cast(),
        dx,
    ))
}

unsafe fn dropout_forward(
    handle: cudnnHandle_t,
    dropoutDesc: cudnnDropoutDescriptor_t,
    xdesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    ydesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    reserveSpace: *mut ::std::os::raw::c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
    to_cudnn(miopenDropoutForward(
        handle.cast(),
        dropoutDesc.cast(),
        ptr::null_mut(),
        xdesc.cast(),
        x,
        ydesc.cast(),
        y,
        reserveSpace,
        reserveSpaceSizeInBytes,
    ))
}

unsafe fn get_dropout_descriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: *mut f32,
    states: *mut *mut ::std::os::raw::c_void,
    seed: *mut ::std::os::raw::c_ulonglong,
) -> cudnnStatus_t {
    let mut use_mask = false;
    let mut state_evo = false;
    let mut rng_mode = miopenRNGType_t::MIOPEN_RNG_PSEUDO_XORWOW;
    to_cudnn(miopenGetDropoutDescriptor(
        dropoutDesc.cast(),
        handle.cast(),
        dropout,
        states,
        seed,
        &mut use_mask,
        &mut state_evo,
        &mut rng_mode,
    ))
}

unsafe fn restore_dropout_descriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::std::os::raw::c_void,
    stateSizeInBytes: usize,
    seed: ::std::os::raw::c_ulonglong,
) -> cudnnStatus_t {
    let use_mask = false;
    let state_evo = false;
    let rng_mode = miopenRNGType_t::MIOPEN_RNG_PSEUDO_XORWOW;
    to_cudnn(miopenRestoreDropoutDescriptor(
        dropoutDesc.cast(),
        handle.cast(),
        dropout,
        states,
        stateSizeInBytes,
        seed,
        use_mask,
        state_evo,
        rng_mode,
    ))
}

unsafe fn set_dropout_descriptor(
    dropoutDesc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut ::std::os::raw::c_void,
    stateSizeInBytes: usize,
    seed: ::std::os::raw::c_ulonglong,
) -> cudnnStatus_t {
    let use_mask = false;
    let state_evo = false;
    let rng_mode = miopenRNGType_t::MIOPEN_RNG_PSEUDO_XORWOW;
    to_cudnn(miopenSetDropoutDescriptor(
        dropoutDesc.cast(),
        handle.cast(),
        dropout,
        states,
        stateSizeInBytes,
        seed,
        use_mask,
        state_evo,
        rng_mode,
    ))
}

unsafe fn dropout_get_reserve_space_size(
    xdesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenDropoutGetReserveSpaceSize(xdesc.cast(), sizeInBytes))
}

unsafe fn dropout_get_states_size(handle: cudnnHandle_t, sizeInBytes: *mut usize) -> cudnnStatus_t {
    to_cudnn(miopenDropoutGetStatesSize(handle.cast(), sizeInBytes))
}

unsafe fn create_dropout_descriptor(dropoutDesc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenCreateDropoutDescriptor(dropoutDesc.cast()))
}

unsafe fn destroy_dropout_descriptor(dropoutDesc: cudnnDropoutDescriptor_t) -> cudnnStatus_t {
    to_cudnn(miopenDestroyDropoutDescriptor(dropoutDesc.cast()))
}

unsafe fn batch_normalization_forward_inference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: *const ::std::os::raw::c_void,
    bnBias: *const ::std::os::raw::c_void,
    estimatedMean: *const ::std::os::raw::c_void,
    estimatedVariance: *const ::std::os::raw::c_void,
    epsilon: f64,
) -> cudnnStatus_t {
    let bat_mode = to_batch_mode(mode);
    to_cudnn(miopenBatchNormalizationForwardInference(
        handle.cast(),
        bat_mode,
        alpha as _,
        beta as _,
        xDesc.cast(),
        x,
        yDesc.cast(),
        y,
        bnScaleBiasMeanVarDesc.cast(),
        bnScale as _,
        bnBias as _,
        estimatedMean as _,
        estimatedVariance as _,
        epsilon,
    ))
}

unsafe fn derive_bn_tensor_descriptor(
    derivedBnDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
) -> cudnnStatus_t {
    let batch_mode = to_batch_mode(mode);
    to_cudnn(miopenDeriveBNTensorDescriptor(
        derivedBnDesc.cast(),
        xDesc.cast(),
        batch_mode,
    ))
}

unsafe fn get_lrn_descriptor(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: *mut ::std::os::raw::c_uint,
    lrnAlpha: *mut f64,
    lrnBeta: *mut f64,
    lrnK: *mut f64,
) -> cudnnStatus_t {
    let mut mode = miopenLRNMode_t::miopenLRNCrossChannel;
    to_cudnn(miopenGetLRNDescriptor(
        normDesc.cast(),
        &mut mode,
        lrnN,
        lrnAlpha,
        lrnBeta,
        lrnK,
    ))
}

unsafe fn get_pooling2d_forward_output_dim(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    n: *mut ::std::os::raw::c_int,
    c: *mut ::std::os::raw::c_int,
    h: *mut ::std::os::raw::c_int,
    w: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    to_cudnn(miopenGetPoolingForwardOutputDim(
        poolingDesc.cast(),
        inputTensorDesc.cast(),
        n,
        c,
        h,
        w,
    ))
}

unsafe fn get_pooling_nd_descriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    nbDimsRequested: ::std::os::raw::c_int,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    nbDims: *mut ::std::os::raw::c_int,
    windowDimA: *mut ::std::os::raw::c_int,
    paddingA: *mut ::std::os::raw::c_int,
    strideA: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    let mut pool_mode = miopenPoolingMode_t::miopenPoolingAverage;
    let status = miopenGetNdPoolingDescriptor(
        poolingDesc.cast(),
        nbDimsRequested,
        &mut pool_mode,
        nbDims,
        windowDimA,
        paddingA,
        strideA,
    );
    *mode = miopen_to_pooling_mode(pool_mode);
    *maxpoolingNanOpt = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
    to_cudnn(status)
}

unsafe fn get_pooling_2d_descriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    windowHeight: *mut ::std::os::raw::c_int,
    windowWidth: *mut ::std::os::raw::c_int,
    verticalPadding: *mut ::std::os::raw::c_int,
    horizontalPadding: *mut ::std::os::raw::c_int,
    verticalStride: *mut ::std::os::raw::c_int,
    horizontalStride: *mut ::std::os::raw::c_int,
) -> cudnnStatus_t {
    let mut pool_mode = miopenPoolingMode_t::miopenPoolingAverage;
    let status = miopenGet2dPoolingDescriptor(
        poolingDesc.cast(),
        &mut pool_mode,
        windowHeight,
        windowWidth,
        verticalPadding,
        horizontalPadding,
        verticalStride,
        horizontalStride,
    );
    *mode = miopen_to_pooling_mode(pool_mode);
    *maxpoolingNanOpt = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
    to_cudnn(status)
}

unsafe fn set_pooling_2d_descriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    windowHeight: ::std::os::raw::c_int,
    windowWidth: ::std::os::raw::c_int,
    verticalPadding: ::std::os::raw::c_int,
    horizontalPadding: ::std::os::raw::c_int,
    verticalStride: ::std::os::raw::c_int,
    horizontalStride: ::std::os::raw::c_int,
) -> cudnnStatus_t {
    let pool_mode = to_pooling_mode(mode);
    to_cudnn(miopenSet2dPoolingDescriptor(
        poolingDesc.cast(),
        pool_mode,
        windowHeight,
        windowWidth,
        verticalPadding,
        horizontalPadding,
        verticalStride,
        horizontalStride,
    ))
}

unsafe fn scale_tensor(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    alpha: *const ::std::os::raw::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenScaleTensor(handle.cast(), yDesc.cast(), y, alpha))
}

unsafe fn set_tensor(
    handle: cudnnHandle_t,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    valuePtr: *const ::std::os::raw::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenSetTensor(handle.cast(), yDesc.cast(), y, valuePtr))
}

unsafe fn reduce_tensor(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    indices: *mut ::std::os::raw::c_void,
    indicesSizeInBytes: usize,
    workspace: *mut ::std::os::raw::c_void,
    workspaceSizeInBytes: usize,
    alpha: *const ::std::os::raw::c_void,
    aDesc: cudnnTensorDescriptor_t,
    A: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    cDesc: cudnnTensorDescriptor_t,
    C: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    to_cudnn(miopenReduceTensor(
        handle.cast(),
        reduceTensorDesc.cast(),
        indices,
        indicesSizeInBytes,
        workspace,
        workspaceSizeInBytes,
        alpha,
        aDesc.cast(),
        A,
        beta,
        cDesc.cast(),
        C,
    ))
}

unsafe fn get_reduction_workspace_size(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenGetReductionWorkspaceSize(
        handle.cast(),
        reduceTensorDesc.cast(),
        aDesc.cast(),
        cDesc.cast(),
        sizeInBytes,
    ))
}

unsafe fn get_reduction_indices_size(
    handle: cudnnHandle_t,
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    aDesc: cudnnTensorDescriptor_t,
    cDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
    to_cudnn(miopenGetReductionIndicesSize(
        handle.cast(),
        reduceTensorDesc.cast(),
        aDesc.cast(),
        cDesc.cast(),
        sizeInBytes,
    ))
}

unsafe fn destroy_reduce_tensor_descriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopenDestroyReduceTensorDescriptor(reduceTensorDesc.cast()))
}

unsafe fn get_reduce_tensor_descriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: *mut cudnnReduceTensorOp_t,
    reduceTensorCompType: *mut cudnnDataType_t,
    reduceTensorNanOpt: *mut cudnnNanPropagation_t,
    reduceTensorIndices: *mut cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: *mut cudnnIndicesType_t,
) -> cudnnStatus_t {
    let mut op = miopenReduceTensorOp_t::MIOPEN_REDUCE_TENSOR_ADD;
    let mut type_ = miopenDataType_t::miopenFloat;
    let mut nan_opt = miopenNanPropagation_t::MIOPEN_NOT_PROPAGATE_NAN;
    let mut indices = miopenReduceTensorIndices_t::MIOPEN_REDUCE_TENSOR_NO_INDICES;
    let mut indices_type = miopenIndicesType_t::MIOPEN_32BIT_INDICES;
    let status = miopenGetReduceTensorDescriptor(
        reduceTensorDesc.cast(),
        &mut op,
        &mut type_,
        &mut nan_opt,
        &mut indices,
        &mut indices_type,
    );
    *reduceTensorOp = cudnnReduceTensorOp_t(op.0);
    *reduceTensorCompType = miopen_datatype_to_cudnn(type_);
    *reduceTensorNanOpt = cudnnNanPropagation_t(nan_opt.0);
    *reduceTensorIndices = cudnnReduceTensorIndices_t(indices.0);
    *reduceTensorIndicesType = cudnnIndicesType_t(indices_type.0);
    to_cudnn(status)
}

pub unsafe extern "system" fn set_reduce_tensor_descriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: cudnnReduceTensorOp_t,
    reduceTensorCompType: cudnnDataType_t,
    reduceTensorNanOpt: cudnnNanPropagation_t,
    reduceTensorIndices: cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: cudnnIndicesType_t,
) -> cudnnStatus_t {
    let op = miopenReduceTensorOp_t(reduceTensorOp.0);
    let type_ = from_data_type(reduceTensorCompType);
    let nan_opt = miopenNanPropagation_t(reduceTensorNanOpt.0);
    let indices = miopenReduceTensorIndices_t(reduceTensorIndices.0);
    let indices_type = miopenIndicesType_t(reduceTensorIndicesType.0);
    to_cudnn(miopenSetReduceTensorDescriptor(
        reduceTensorDesc.cast(),
        op,
        type_,
        nan_opt,
        indices,
        indices_type,
    ))
}

unsafe fn create_reduce_tensor_descriptor(
    reduceTensorDesc: *mut cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t {
    to_cudnn(miopenCreateReduceTensorDescriptor(reduceTensorDesc.cast()))
}