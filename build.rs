use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders/keccak.comp");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = fs::read_to_string("shaders/keccak.comp")
        .expect("missing shaders/keccak.comp");

    let compiler = shaderc::Compiler::new().unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_optimization_level(shaderc::OptimizationLevel::Zero); // no opts — safer on buggy drivers
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );

    let artifact = compiler
        .compile_into_spirv(&src, shaderc::ShaderKind::Compute, "keccak.comp", "main", Some(&opts))
        .expect("shader compile failed");

    fs::write(out_dir.join("keccak.spv"), artifact.as_binary_u8()).unwrap();
}