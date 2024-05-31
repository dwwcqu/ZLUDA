fn main() {
    println!("cargo:rustc-link-lib=dylib=hipsolver");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
}
