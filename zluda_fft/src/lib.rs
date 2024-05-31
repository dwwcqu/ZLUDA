#![allow(non_snake_case)]
#[allow(warnings)]
mod cufft;
pub use cufft::*;

#[allow(warnings)]
mod cufftxt;
pub use cufftxt::*;

use cuda_types::*;
use hipfft_sys::*;
use lazy_static::lazy_static;
use slab::Slab;
use std::{mem, ptr, sync::Mutex};

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> cufftResult {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> cufftResult {
    cufftResult::CUFFT_NOT_SUPPORTED
}

lazy_static! {
    static ref PLANS: Mutex<Slab<Plan>> = Mutex::new(Slab::new());
}

struct Plan(hipfftHandle);
unsafe impl Send for Plan {}

unsafe fn create(handle: *mut cufftHandle) -> cufftResult {
    let mut hip_handle = unsafe { mem::zeroed() };
    let error = hipfftCreate(&mut hip_handle);
    if error != hipfftResult::HIPFFT_SUCCESS {
        return cufftResult::CUFFT_INTERNAL_ERROR;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan(hip_handle))
    };
    *handle = plan_key as i32;
    cufftResult::CUFFT_SUCCESS
}

fn destroy(plan: i32) -> cufftResult_t {
    let mut plans = PLANS.lock().unwrap();
    plans.remove(plan as usize);
    cufftResult::CUFFT_SUCCESS
}

unsafe fn make_plan_many_64(
    plan: i32,
    rank: i32,
    n: *mut i64,
    inembed: *mut i64,
    istride: i64,
    idist: i64,
    onembed: *mut i64,
    ostride: i64,
    odist: i64,
    type_: cufftType,
    batch: i64,
    work_size: *mut usize,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let type_ = cuda_type(type_);
    to_cuda(hipfftMakePlanMany64(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
        work_size,
    ))
}

fn cuda_type(type_: cufftType) -> hipfftType_t {
    match type_ {
        cufftType::CUFFT_R2C => hipfftType_t::HIPFFT_R2C,
        cufftType::CUFFT_C2R => hipfftType_t::HIPFFT_C2R,
        cufftType::CUFFT_C2C => hipfftType_t::HIPFFT_C2C,
        cufftType::CUFFT_D2Z => hipfftType_t::HIPFFT_D2Z,
        cufftType::CUFFT_Z2D => hipfftType_t::HIPFFT_Z2D,
        cufftType::CUFFT_Z2Z => hipfftType_t::HIPFFT_Z2Z,
        _ => panic!(),
    }
}

fn get_hip_plan(plan: cufftHandle) -> Result<hipfftHandle, cufftResult_t> {
    let plans = PLANS.lock().unwrap();
    plans
        .get(plan as usize)
        .map(|p| p.0)
        .ok_or(cufftResult_t::CUFFT_INVALID_PLAN)
}

fn to_cuda(result: hipfftResult) -> cufftResult_t {
    match result {
        hipfftResult::HIPFFT_SUCCESS => cufftResult_t::CUFFT_SUCCESS,
        _ => cufftResult_t::CUFFT_INTERNAL_ERROR,
    }
}

unsafe fn make_plan_many(
    plan: i32,
    rank: i32,
    n: *mut i32,
    inembed: *mut i32,
    istride: i32,
    idist: i32,
    onembed: *mut i32,
    ostride: i32,
    odist: i32,
    type_: cufftType,
    batch: i32,
    work_size: *mut usize,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let type_ = cuda_type(type_);
    to_cuda(hipfftMakePlanMany(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
        work_size,
    ))
}

unsafe fn plan_many(
    plan: *mut i32,
    rank: i32,
    n: *mut i32,
    inembed: *mut i32,
    istride: i32,
    idist: i32,
    onembed: *mut i32,
    ostride: i32,
    odist: i32,
    type_: cufftType,
    batch: i32,
) -> cufftResult_t {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlanMany(
        &mut hip_plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
    ));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan(hip_plan))
    };
    *plan = plan_key as i32;
    result
}

unsafe fn set_stream(plan: i32, stream: *mut cufft::CUstream_st) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_get_export_table = lib
        .get::<unsafe extern "C" fn(
            ppExportTable: *mut *const ::std::os::raw::c_void,
            pExportTableId: *const CUuuid,
        ) -> CUresult>(b"cuGetExportTable\0")
        .unwrap();
    let mut export_table = ptr::null();
    let error = (cu_get_export_table)(&mut export_table, &zluda_dark_api::ZludaExt::GUID);
    assert_eq!(error, CUresult::CUDA_SUCCESS);
    let zluda_ext = zluda_dark_api::ZludaExt::new(export_table);
    let stream: Result<_, _> = zluda_ext.get_hip_stream(stream as _).into();
    to_cuda(hipfftSetStream(hip_plan, stream.unwrap() as _))
}

unsafe fn exec_c2c(
    plan: i32,
    idata: *mut cufft::float2,
    odata: *mut cufft::float2,
    direction: i32,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecC2C(
        hip_plan,
        idata.cast(),
        odata.cast(),
        direction,
    ))
}

unsafe fn exec_z2z(
    plan: i32,
    idata: *mut cufft::double2,
    odata: *mut cufft::double2,
    direction: i32,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecZ2Z(
        hip_plan,
        idata.cast(),
        odata.cast(),
        direction,
    ))
}

unsafe fn exec_r2c(plan: i32, idata: *mut f32, odata: *mut cufft::float2) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecR2C(hip_plan, idata, odata.cast()))
}

unsafe fn exec_c2r(plan: i32, idata: *mut cufft::float2, odata: *mut f32) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecC2R(hip_plan, idata.cast(), odata))
}

unsafe fn plan_3d(plan: *mut i32, nx: i32, ny: i32, nz: i32, type_: cufftType) -> cufftResult {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlan3d(&mut hip_plan, nx, ny, nz, type_));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan(hip_plan))
    };
    *plan = plan_key as i32;
    result
}

unsafe fn plan_1d(
    plan: *mut cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
) -> cufftResult {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlan1d(&mut hip_plan, nx, type_, batch));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan(hip_plan))
    };
    *plan = plan_key as i32;
    result
}

unsafe fn make_plan_1d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };

    to_cuda(hipfftMakePlan1d(hip_plan, nx, data_type, batch, workSize))
}

unsafe fn plan_2d(plan: *mut i32, nx: i32, ny: i32, type_: cufftType) -> cufftResult {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlan2d(&mut hip_plan, nx, ny, type_));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan(hip_plan))
    };
    *plan = plan_key as i32;
    result
}

unsafe fn make_plan_2d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };

    to_cuda(hipfftMakePlan2d(hip_plan, nx, ny, data_type, workSize))
}

unsafe fn make_plan_3d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };

    to_cuda(hipfftMakePlan3d(hip_plan, nx, ny, nz, data_type, workSize))
}

unsafe fn get_size_many_64(
    plan: cufftHandle,
    rank: ::std::os::raw::c_int,
    n: *mut ::std::os::raw::c_longlong,
    inembed: *mut ::std::os::raw::c_longlong,
    istride: ::std::os::raw::c_longlong,
    idist: ::std::os::raw::c_longlong,
    onembed: *mut ::std::os::raw::c_longlong,
    ostride: ::std::os::raw::c_longlong,
    odist: ::std::os::raw::c_longlong,
    type_: cufftType,
    batch: ::std::os::raw::c_longlong,
    work_size: *mut usize,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let data_type = cuda_type(type_);
    to_cuda(hipfftGetSizeMany64(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, data_type, batch, work_size,
    ))
}

unsafe fn estimate_1d(
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    to_cuda(hipfftEstimate1d(nx, data_type, batch, workSize))
}

unsafe fn estimate_2d(
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    to_cuda(hipfftEstimate2d(nx, ny, data_type, workSize))
}

unsafe fn estimate_3d(
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    to_cuda(hipfftEstimate3d(nx, ny, nz, data_type, workSize))
}

unsafe fn estimate_many(
    rank: ::std::os::raw::c_int,
    n: *mut ::std::os::raw::c_int,
    inembed: *mut ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    onembed: *mut ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    to_cuda(hipfftEstimateMany(
        rank, n, inembed, istride, idist, onembed, ostride, odist, data_type, batch, workSize,
    ))
}

unsafe fn get_size_1d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(handle) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftGetSize1d(hip_plan, nx, data_type, batch, workSize))
}

unsafe fn get_size_2d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(handle) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftGetSize2d(hip_plan, nx, ny, data_type, workSize))
}

unsafe fn get_size_3d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(handle) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftGetSize3d(hip_plan, nx, ny, nz, data_type, workSize))
}

unsafe fn get_size_many(
    handle: cufftHandle,
    rank: ::std::os::raw::c_int,
    n: *mut ::std::os::raw::c_int,
    inembed: *mut ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    onembed: *mut ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workArea: *mut usize,
) -> cufftResult {
    let data_type = cuda_type(type_);
    let hip_plan = match get_hip_plan(handle) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftGetSizeMany(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, data_type, batch,
        workArea,
    ))
}

unsafe fn get_size(handle: cufftHandle, workSize: *mut usize) -> cufftResult {
    let hip_plan = match get_hip_plan(handle) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftGetSize(hip_plan, workSize))
}

unsafe fn set_work_area(plan: cufftHandle, workArea: *mut ::std::os::raw::c_void) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftSetWorkArea(hip_plan, workArea))
}

unsafe fn set_auto_allocation(
    plan: cufftHandle,
    autoAllocate: ::std::os::raw::c_int,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftSetAutoAllocation(hip_plan, autoAllocate))
}

unsafe fn exec_d2z(
    plan: cufftHandle,
    idata: *mut cufftDoubleReal,
    odata: *mut cufftDoubleComplex,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecD2Z(hip_plan, idata.cast(), odata.cast()))
}

unsafe fn exec_z2d(
    plan: cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleReal,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecZ2D(hip_plan, idata.cast(), odata.cast()))
}

unsafe fn get_version(version: *mut ::std::os::raw::c_int) -> cufftResult {
    to_cuda(hipfftGetVersion(version))
}

unsafe fn get_property(
    type_: libraryPropertyType,
    value: *mut ::std::os::raw::c_int,
) -> cufftResult {
    let lib_type = hipfftLibraryPropertyType(type_.0);
    to_cuda(hipfftGetProperty(lib_type, value))
}
