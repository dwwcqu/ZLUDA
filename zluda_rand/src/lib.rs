#![allow(non_snake_case)]
#[allow(warnings)]
mod curand;
pub use curand::*;

use hiprand_sys::*;

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> curandStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> curandStatus_t {
    curandStatus_t::CURAND_STATUS_INTERNAL_ERROR
}

fn to_cuda(status: hiprandStatus_t) -> curandStatus_t {
    match status {
        hiprandStatus_t::HIPRAND_STATUS_SUCCESS => curandStatus_t::CURAND_STATUS_SUCCESS,
        hiprandStatus_t::HIPRAND_STATUS_NOT_INITIALIZED => {
            curandStatus_t::CURAND_STATUS_NOT_INITIALIZED
        }
        hiprandStatus_t::HIPRAND_STATUS_TYPE_ERROR => curandStatus_t::CURAND_STATUS_TYPE_ERROR,
        hiprandStatus_t::HIPRAND_STATUS_ALLOCATION_FAILED => {
            curandStatus_t::CURAND_STATUS_ALLOCATION_FAILED
        }
        hiprandStatus_t::HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED => {
            curandStatus_t::CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
        }
        hiprandStatus_t::HIPRAND_STATUS_INITIALIZATION_FAILED => {
            curandStatus_t::CURAND_STATUS_INITIALIZATION_FAILED
        }
        hiprandStatus_t::HIPRAND_STATUS_INTERNAL_ERROR => {
            curandStatus_t::CURAND_STATUS_INTERNAL_ERROR
        }
        hiprandStatus_t::HIPRAND_STATUS_LAUNCH_FAILURE => {
            curandStatus_t::CURAND_STATUS_LAUNCH_FAILURE
        }
        hiprandStatus_t::HIPRAND_STATUS_LENGTH_NOT_MULTIPLE => {
            curandStatus_t::CURAND_STATUS_LENGTH_NOT_MULTIPLE
        }
        hiprandStatus_t::HIPRAND_STATUS_NOT_IMPLEMENTED => {
            curandStatus_t::CURAND_STATUS_INTERNAL_ERROR
        }
        hiprandStatus_t::HIPRAND_STATUS_OUT_OF_RANGE => curandStatus_t::CURAND_STATUS_OUT_OF_RANGE,
        hiprandStatus_t::HIPRAND_STATUS_VERSION_MISMATCH => {
            curandStatus_t::CURAND_STATUS_VERSION_MISMATCH
        }
        hiprandStatus_t::HIPRAND_STATUS_ARCH_MISMATCH => {
            curandStatus_t::CURAND_STATUS_ARCH_MISMATCH
        }
        hiprandStatus_t::HIPRAND_STATUS_PREEXISTING_FAILURE => {
            curandStatus_t::CURAND_STATUS_PREEXISTING_FAILURE
        }
        _ => panic!(),
    }
}

fn rng_from_cuda(rng_type: curandRngType_t) -> hiprandRngType_t {
    match rng_type {
        curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT => hiprandRngType::HIPRAND_RNG_PSEUDO_DEFAULT,
        curandRngType_t::CURAND_RNG_PSEUDO_MRG32K3A => hiprandRngType::HIPRAND_RNG_PSEUDO_MRG32K3A,
        curandRngType_t::CURAND_RNG_PSEUDO_MT19937 => hiprandRngType::HIPRAND_RNG_PSEUDO_MT19937,
        curandRngType_t::CURAND_RNG_PSEUDO_MTGP32 => hiprandRngType::HIPRAND_RNG_PSEUDO_MTGP32,
        curandRngType_t::CURAND_RNG_PSEUDO_PHILOX4_32_10 => {
            hiprandRngType::HIPRAND_RNG_PSEUDO_PHILOX4_32_10
        }
        curandRngType_t::CURAND_RNG_PSEUDO_XORWOW => hiprandRngType::HIPRAND_RNG_PSEUDO_XORWOW,
        curandRngType_t::CURAND_RNG_QUASI_DEFAULT => hiprandRngType::HIPRAND_RNG_QUASI_DEFAULT,
        curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 => {
            hiprandRngType::HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
        }
        curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 => {
            hiprandRngType::HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
        }
        curandRngType_t::CURAND_RNG_QUASI_SOBOL32 => hiprandRngType::HIPRAND_RNG_QUASI_SOBOL32,
        curandRngType_t::CURAND_RNG_QUASI_SOBOL64 => hiprandRngType::HIPRAND_RNG_QUASI_SOBOL64,
        curandRngType_t::CURAND_RNG_TEST => hiprandRngType::HIPRAND_RNG_TEST,
        _ => panic!(),
    }
}

unsafe fn create_generator(
    generator: *mut curandGenerator_t,
    rng_type: curandRngType_t,
) -> curandStatus_t {
    let rng = rng_from_cuda(rng_type);
    to_cuda(hiprandCreateGenerator(generator.cast(), rng))
}

unsafe fn create_generator_host(
    generator: *mut curandGenerator_t,
    rng_type: curandRngType_t,
) -> curandStatus_t {
    let rng = rng_from_cuda(rng_type);
    to_cuda(hiprandCreateGeneratorHost(generator.cast(), rng))
}

unsafe fn destroy_generator(generator: curandGenerator_t) -> curandStatus_t {
    to_cuda(hiprandDestroyGenerator(generator.cast()))
}

unsafe fn get_version(version: *mut ::std::os::raw::c_int) -> curandStatus_t {
    to_cuda(hiprandGetVersion(version))
}

unsafe fn set_stream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t {
    to_cuda(hiprandSetStream(generator.cast(), stream.cast()))
}

unsafe fn set_pseudo_random_generator_seed(
    generator: curandGenerator_t,
    seed: ::std::os::raw::c_ulonglong,
) -> curandStatus_t {
    to_cuda(hiprandSetPseudoRandomGeneratorSeed(generator.cast(), seed))
}

unsafe fn set_generator_offset(
    generator: curandGenerator_t,
    offset: ::std::os::raw::c_ulonglong,
) -> curandStatus_t {
    to_cuda(hiprandSetGeneratorOffset(generator.cast(), offset))
}

unsafe fn set_quasi_random_generator_dimensions(
    generator: curandGenerator_t,
    num_dimensions: ::std::os::raw::c_uint,
) -> curandStatus_t {
    to_cuda(hiprandSetQuasiRandomGeneratorDimensions(
        generator.cast(),
        num_dimensions,
    ))
}

unsafe fn generate(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    num: usize,
) -> curandStatus_t {
    to_cuda(hiprandGenerate(generator.cast(), outputPtr, num))
}

unsafe fn generate_uniform(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    num: usize,
) -> curandStatus_t {
    to_cuda(hiprandGenerateUniform(generator.cast(), outputPtr, num))
}

unsafe fn generate_uniform_double(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    num: usize,
) -> curandStatus_t {
    to_cuda(hiprandGenerateUniformDouble(
        generator.cast(),
        outputPtr,
        num,
    ))
}

unsafe fn generate_normal(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t {
    to_cuda(hiprandGenerateNormal(
        generator.cast(),
        outputPtr,
        n,
        mean,
        stddev,
    ))
}

unsafe fn generate_normal_double(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t {
    to_cuda(hiprandGenerateNormalDouble(
        generator.cast(),
        outputPtr,
        n,
        mean,
        stddev,
    ))
}

unsafe fn generate_log_normal(
    generator: curandGenerator_t,
    outputPtr: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t {
    to_cuda(hiprandGenerateLogNormal(
        generator.cast(),
        outputPtr,
        n,
        mean,
        stddev,
    ))
}

unsafe fn generate_log_normal_double(
    generator: curandGenerator_t,
    outputPtr: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t {
    to_cuda(hiprandGenerateLogNormalDouble(
        generator.cast(),
        outputPtr,
        n,
        mean,
        stddev,
    ))
}

unsafe fn create_poisson_distribution(
    lambda: f64,
    discrete_distribution: *mut curandDiscreteDistribution_t,
) -> curandStatus_t {
    to_cuda(hiprandCreatePoissonDistribution(
        lambda,
        discrete_distribution.cast(),
    ))
}

unsafe fn destroy_distribution(
    discrete_distribution: curandDiscreteDistribution_t,
) -> curandStatus_t {
    to_cuda(hiprandDestroyDistribution(discrete_distribution.cast()))
}

unsafe fn generate_poisson(
    generator: curandGenerator_t,
    outputPtr: *mut ::std::os::raw::c_uint,
    n: usize,
    lambda: f64,
) -> curandStatus_t {
    to_cuda(hiprandGeneratePoisson(generator.cast(), outputPtr, n, lambda))
}

unsafe fn generate_seeds(generator: curandGenerator_t) -> curandStatus_t {
    to_cuda(hiprandGenerateSeeds(generator.cast()))
}