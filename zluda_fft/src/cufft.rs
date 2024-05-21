/* automatically generated by rust-bindgen 0.66.1 */

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
pub type cuFloatComplex = float2;
pub type cuDoubleComplex = double2;
pub type cuComplex = cuFloatComplex;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[doc = " CUDA stream"]
pub type cudaStream_t = *mut CUstream_st;
impl libraryPropertyType_t {
    pub const MAJOR_VERSION: libraryPropertyType_t = libraryPropertyType_t(0);
}
impl libraryPropertyType_t {
    pub const MINOR_VERSION: libraryPropertyType_t = libraryPropertyType_t(1);
}
impl libraryPropertyType_t {
    pub const PATCH_LEVEL: libraryPropertyType_t = libraryPropertyType_t(2);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct libraryPropertyType_t(pub ::std::os::raw::c_uint);
pub use self::libraryPropertyType_t as libraryPropertyType;
impl cufftResult_t {
    pub const CUFFT_SUCCESS: cufftResult_t = cufftResult_t(0);
}
impl cufftResult_t {
    pub const CUFFT_INVALID_PLAN: cufftResult_t = cufftResult_t(1);
}
impl cufftResult_t {
    pub const CUFFT_ALLOC_FAILED: cufftResult_t = cufftResult_t(2);
}
impl cufftResult_t {
    pub const CUFFT_INVALID_TYPE: cufftResult_t = cufftResult_t(3);
}
impl cufftResult_t {
    pub const CUFFT_INVALID_VALUE: cufftResult_t = cufftResult_t(4);
}
impl cufftResult_t {
    pub const CUFFT_INTERNAL_ERROR: cufftResult_t = cufftResult_t(5);
}
impl cufftResult_t {
    pub const CUFFT_EXEC_FAILED: cufftResult_t = cufftResult_t(6);
}
impl cufftResult_t {
    pub const CUFFT_SETUP_FAILED: cufftResult_t = cufftResult_t(7);
}
impl cufftResult_t {
    pub const CUFFT_INVALID_SIZE: cufftResult_t = cufftResult_t(8);
}
impl cufftResult_t {
    pub const CUFFT_UNALIGNED_DATA: cufftResult_t = cufftResult_t(9);
}
impl cufftResult_t {
    pub const CUFFT_INCOMPLETE_PARAMETER_LIST: cufftResult_t = cufftResult_t(10);
}
impl cufftResult_t {
    pub const CUFFT_INVALID_DEVICE: cufftResult_t = cufftResult_t(11);
}
impl cufftResult_t {
    pub const CUFFT_PARSE_ERROR: cufftResult_t = cufftResult_t(12);
}
impl cufftResult_t {
    pub const CUFFT_NO_WORKSPACE: cufftResult_t = cufftResult_t(13);
}
impl cufftResult_t {
    pub const CUFFT_NOT_IMPLEMENTED: cufftResult_t = cufftResult_t(14);
}
impl cufftResult_t {
    pub const CUFFT_LICENSE_ERROR: cufftResult_t = cufftResult_t(15);
}
impl cufftResult_t {
    pub const CUFFT_NOT_SUPPORTED: cufftResult_t = cufftResult_t(16);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cufftResult_t(pub ::std::os::raw::c_uint);
pub use self::cufftResult_t as cufftResult;
pub type cufftReal = f32;
pub type cufftDoubleReal = f64;
pub type cufftComplex = cuComplex;
pub type cufftDoubleComplex = cuDoubleComplex;
impl cufftType_t {
    pub const CUFFT_R2C: cufftType_t = cufftType_t(42);
}
impl cufftType_t {
    pub const CUFFT_C2R: cufftType_t = cufftType_t(44);
}
impl cufftType_t {
    pub const CUFFT_C2C: cufftType_t = cufftType_t(41);
}
impl cufftType_t {
    pub const CUFFT_D2Z: cufftType_t = cufftType_t(106);
}
impl cufftType_t {
    pub const CUFFT_Z2D: cufftType_t = cufftType_t(108);
}
impl cufftType_t {
    pub const CUFFT_Z2Z: cufftType_t = cufftType_t(105);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cufftType_t(pub ::std::os::raw::c_uint);
pub use self::cufftType_t as cufftType;
pub type cufftHandle = ::std::os::raw::c_int;

#[no_mangle]
pub unsafe extern "system" fn cufftPlan1d(
    plan: *mut cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
) -> cufftResult {
    crate::plan_1d(plan, nx, type_, batch)
}

#[no_mangle]
pub unsafe extern "system" fn cufftPlan2d(
    plan: *mut cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftPlan3d(
    plan: *mut cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
) -> cufftResult {
    crate::plan_3d(plan, nx, ny, nz, type_)
}

#[no_mangle]
pub unsafe extern "system" fn cufftPlanMany(
    plan: *mut cufftHandle,
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
) -> cufftResult {
    crate::plan_many(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
    )
}

#[no_mangle]
pub unsafe extern "system" fn cufftMakePlan1d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftMakePlan2d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftMakePlan3d(
    plan: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftMakePlanMany(
    plan: cufftHandle,
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
    crate::make_plan_many(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
    )
}

#[no_mangle]
pub unsafe extern "system" fn cufftMakePlanMany64(
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
    workSize: *mut usize,
) -> cufftResult {
    crate::make_plan_many_64(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
    )
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSizeMany64(
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
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftEstimate1d(
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftEstimate2d(
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftEstimate3d(
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftEstimateMany(
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
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftCreate(handle: *mut cufftHandle) -> cufftResult {
    crate::create(handle)
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSize1d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    type_: cufftType,
    batch: ::std::os::raw::c_int,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSize2d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSize3d(
    handle: cufftHandle,
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    nz: ::std::os::raw::c_int,
    type_: cufftType,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSizeMany(
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
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetSize(
    handle: cufftHandle,
    workSize: *mut usize,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftSetWorkArea(
    plan: cufftHandle,
    workArea: *mut ::std::os::raw::c_void,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftSetAutoAllocation(
    plan: cufftHandle,
    autoAllocate: ::std::os::raw::c_int,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecC2C(
    plan: cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftComplex,
    direction: ::std::os::raw::c_int,
) -> cufftResult {
    crate::exec_c2c(plan, idata, odata, direction)
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecR2C(
    plan: cufftHandle,
    idata: *mut cufftReal,
    odata: *mut cufftComplex,
) -> cufftResult {
    crate::exec_r2c(plan, idata, odata)
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecC2R(
    plan: cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftReal,
) -> cufftResult {
    crate::exec_c2r(plan, idata, odata)
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecZ2Z(
    plan: cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleComplex,
    direction: ::std::os::raw::c_int,
) -> cufftResult {
    crate::exec_z2z(plan, idata, odata, direction)
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecD2Z(
    plan: cufftHandle,
    idata: *mut cufftDoubleReal,
    odata: *mut cufftDoubleComplex,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftExecZ2D(
    plan: cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleReal,
) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftSetStream(
    plan: cufftHandle,
    stream: cudaStream_t,
) -> cufftResult {
    crate::set_stream(plan, stream)
}

#[no_mangle]
pub unsafe extern "system" fn cufftDestroy(plan: cufftHandle) -> cufftResult {
    crate::destroy(plan)
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetVersion(version: *mut ::std::os::raw::c_int) -> cufftResult {
    crate::unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cufftGetProperty(
    type_: libraryPropertyType,
    value: *mut ::std::os::raw::c_int,
) -> cufftResult {
    crate::unsupported()
}
