fn main() {
    println!("cargo:rustc-link-lib=dylib=hiprand");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
}
