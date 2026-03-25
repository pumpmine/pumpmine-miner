use anyhow::{anyhow, Result};
use ash::{vk, Device, Entry, Instance};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use std::ffi::CStr;
use std::io::{self, Write};
use tiny_keccak::{Hasher, Keccak};

// Note: These modules are assumed to exist based on your original code
mod config;
mod miner;
mod token_abi;

use crate::{config::MinerConfig, miner::miner};

// ── CLI Arguments ────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Index of the GPU to use (bypasses selection UI)
    #[arg(long)]
    gpu: Option<usize>,

    /// Token address of mined coin, overrides config
    #[arg(long)]
    token: Option<String>,
}

// ── Shader SPIR-V (compiled at build time) ────────────────────────────────────
const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/keccak.spv"));

// ── GPU result buffer layout (must match shader binding) ─────────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
struct GpuResult {
    found: u32,
    nonce_lo: u32,
    nonce_hi: u32,
    digest: [u32; 8],
}

// ── Push constants (must match shader push_constant block) ───────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Push {
    ch: [u32; 8], // challenge as 8 BE u32s
    sn: [u32; 5], // sender as 5 BE u32s
    base_lo: u32,
    base_hi: u32,
    tg: [u32; 8], // target as 8 BE u32s
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn be_words<const N: usize, const W: usize>(b: &[u8; N]) -> [u32; W] {
    assert_eq!(N, W * 4);
    let mut out = [0u32; W];
    for (i, c) in b.chunks_exact(4).enumerate() {
        out[i] = u32::from_be_bytes(c.try_into().unwrap());
    }
    out
}

pub fn cpu_keccak256(challenge: &[u8; 32], sender: &[u8; 20], nonce: u64) -> [u8; 32] {
    let mut msg = [0u8; 84];
    msg[0..32].copy_from_slice(challenge);
    msg[32..52].copy_from_slice(sender);
    msg[52..84].fill(0);
    msg[52 + 24..84].copy_from_slice(&nonce.to_be_bytes());

    let mut k = Keccak::v256();
    k.update(&msg);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

// ── Vulkan GPU context ────────────────────────────────────────────────────────
pub struct Gpu {
    pub dev: Device,
    _entry: Entry,
    instance: Instance,
    #[allow(dead_code)]
    phys: vk::PhysicalDevice,
    queue: vk::Queue,
    #[allow(dead_code)]
    qfam: u32,
    ds_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    ds_pool: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    shader: vk::ShaderModule,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    res_buf: vk::Buffer,
    res_mem: vk::DeviceMemory,
}

impl Gpu {
    fn new(gpu_override: Option<usize>) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo {
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo {
                    p_application_info: &app_info,
                    ..Default::default()
                },
                None,
            )?
        };

        let phys_devs = unsafe { instance.enumerate_physical_devices()? };
        if phys_devs.is_empty() {
            return Err(anyhow!("no Vulkan GPU found"));
        }

        // Filter devices that support compute
        let mut available_gpus = Vec::new();
        for &pd in &phys_devs {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let qprops = unsafe { instance.get_physical_device_queue_family_properties(pd) };

            if let Some(qfam) = qprops.iter().enumerate().find_map(|(i, q)| {
                if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    Some(i as u32)
                } else {
                    None
                }
            }) {
                let name = unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                available_gpus.push((pd, name, qfam));
            }
        }

        if available_gpus.is_empty() {
            return Err(anyhow!("no GPUs with compute support found"));
        }

        // Selection Logic
        let selected_idx = if let Some(idx) = gpu_override {
            if idx >= available_gpus.len() {
                return Err(anyhow!("Provided GPU index {} is out of range", idx));
            }
            idx
        } else if available_gpus.len() == 1 {
            log::info!("Single GPU found: {}", available_gpus[0].1);
            0
        } else {
            println!("\n--- Select GPU ---");
            for (i, (_, name, _)) in available_gpus.iter().enumerate() {
                println!("[{}] {}", i, name);
            }
            print!("Enter index: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            input
                .trim()
                .parse::<usize>()
                .map_err(|_| anyhow!("Invalid index input"))?
        };

        let (phys, name, qfam) = available_gpus
            .get(selected_idx)
            .ok_or_else(|| anyhow!("Invalid GPU selection"))?;

        println!("Selected GPU: {} (Queue family: {})\n", name, qfam);

        // --- Device Creation ---
        let prios = [1.0f32];
        let q_ci = vk::DeviceQueueCreateInfo {
            queue_family_index: *qfam,
            queue_count: 1,
            p_queue_priorities: prios.as_ptr(),
            ..Default::default()
        };
        let dev = unsafe {
            instance.create_device(
                *phys,
                &vk::DeviceCreateInfo {
                    queue_create_info_count: 1,
                    p_queue_create_infos: &q_ci,
                    ..Default::default()
                },
                None,
            )?
        };
        let queue = unsafe { dev.get_device_queue(*qfam, 0) };

        // --- Pipeline and Resource Setup ---
        let binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };
        let ds_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 1,
                    p_bindings: &binding,
                    ..Default::default()
                },
                None,
            )?
        };

        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 92,
        };
        let set_layouts = [ds_layout];
        let pipe_layout = unsafe {
            dev.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: set_layouts.as_ptr(),
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &push_range,
                    ..Default::default()
                },
                None,
            )?
        };

        let words: Vec<u32> = SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let shader = unsafe {
            dev.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: words.len() * 4,
                    p_code: words.as_ptr(),
                    ..Default::default()
                },
                None,
            )?
        };

        let entry_name = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader,
            p_name: entry_name.as_ptr(),
            ..Default::default()
        };
        let pipeline = unsafe {
            dev.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo {
                    stage,
                    layout: pipe_layout,
                    ..Default::default()
                }],
                None,
            )
            .map_err(|(_, e)| e)?[0]
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        };
        let ds_pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 1,
                    p_pool_sizes: &pool_size,
                    ..Default::default()
                },
                None,
            )?
        };
        let ds = unsafe {
            dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: ds_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                ..Default::default()
            })?[0]
        };

        let res_size = std::mem::size_of::<GpuResult>() as vk::DeviceSize;
        let res_buf = unsafe {
            dev.create_buffer(
                &vk::BufferCreateInfo {
                    size: res_size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?
        };
        let mem_reqs = unsafe { dev.get_buffer_memory_requirements(res_buf) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(*phys) };
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            })
            .ok_or_else(|| anyhow!("no suitable memory type"))?;

        let res_mem = unsafe {
            dev.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: mem_reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?
        };
        unsafe { dev.bind_buffer_memory(res_buf, res_mem, 0)? };

        let buf_info = vk::DescriptorBufferInfo {
            buffer: res_buf,
            offset: 0,
            range: res_size,
        };
        unsafe {
            dev.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    dst_set: ds,
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buf_info,
                    ..Default::default()
                }],
                &[],
            );
        }

        let cmd_pool = unsafe {
            dev.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: *qfam,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ..Default::default()
                },
                None,
            )?
        };
        let cmd_buf = unsafe {
            dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };

        let fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None)? };

        Ok(Self {
            _entry: entry,
            instance,
            phys: *phys,
            dev,
            queue,
            qfam: *qfam,
            ds_layout,
            pipe_layout,
            pipeline,
            ds_pool,
            ds,
            shader,
            cmd_pool,
            cmd_buf,
            fence,
            res_buf,
            res_mem,
        })
    }

    fn dispatch(&self, push: &Push, groups: u32) -> Result<GpuResult> {
        let dev = &self.dev;
        let res_size = std::mem::size_of::<GpuResult>() as vk::DeviceSize;

        unsafe {
            let ptr = dev.map_memory(self.res_mem, 0, res_size, vk::MemoryMapFlags::empty())?;
            std::ptr::write_bytes(ptr as *mut u8, 0, res_size as usize);
            dev.unmap_memory(self.res_mem);
        }

        unsafe {
            dev.begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            dev.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[self.ds],
                &[],
            );
            dev.cmd_push_constants(
                self.cmd_buf,
                self.pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push),
            );
            dev.cmd_dispatch(self.cmd_buf, groups, 1, 1);
            dev.end_command_buffer(self.cmd_buf)?;
        }

        unsafe {
            dev.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &self.cmd_buf,
                    ..Default::default()
                }],
                self.fence,
            )?;
            dev.wait_for_fences(&[self.fence], true, 30_000_000_000)?;
            dev.reset_fences(&[self.fence])?;
            dev.reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;
        }

        let result = unsafe {
            let ptr = dev.map_memory(self.res_mem, 0, res_size, vk::MemoryMapFlags::empty())?;
            let r = std::ptr::read(ptr as *const GpuResult);
            dev.unmap_memory(self.res_mem);
            r
        };

        Ok(result)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
            self.dev.destroy_fence(self.fence, None);
            self.dev.destroy_command_pool(self.cmd_pool, None);
            self.dev.free_memory(self.res_mem, None);
            self.dev.destroy_buffer(self.res_buf, None);
            self.dev.destroy_pipeline(self.pipeline, None);
            self.dev.destroy_pipeline_layout(self.pipe_layout, None);
            self.dev.destroy_descriptor_pool(self.ds_pool, None);
            self.dev.destroy_descriptor_set_layout(self.ds_layout, None);
            self.dev.destroy_shader_module(self.shader, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn mine(
    gpu: &Gpu,
    challenge: &[u8; 32],
    sender: &[u8; 20],
    target: &[u8; 32],
    base: u64,
    batch: u32,
) -> Result<Option<(u64, [u8; 32])>> {
    let mut sn20 = [0u8; 20];
    sn20.copy_from_slice(sender);

    let push = Push {
        ch: be_words(challenge),
        sn: be_words(&sn20),
        base_lo: base as u32,
        base_hi: (base >> 32) as u32,
        tg: be_words(target),
    };

    let groups = batch.div_ceil(256);
    let r = gpu.dispatch(&push, groups)?;

    if r.found == 0 {
        return Ok(None);
    }

    let nonce = ((r.nonce_hi as u64) << 32) | r.nonce_lo as u64;
    let mut digest = [0u8; 32];
    for (i, &w) in r.digest.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
    }
    Ok(Some((nonce, digest)))
}

// ── Entry point ───────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        r#"______                                _             ___  ____
| ___ \                              (_)            |  \/  (_)
| |_/ /   _ _ __ ___  _ __  _ __ ___  _ _ __   ___  | .  . |_ _ __   ___ _ __
|  __/ | | | '_ ` _ \| '_ \| '_ ` _ \| | '_ \ / _ \ | |\/| | | '_ \ / _ \ '__|
| |  | |_| | | | | | | |_) | | | | | | | | | |  __/ | |  | | | | | |  __/ |
\_|   \__,_|_| |_| |_| .__/|_| |_| |_|_|_| |_|\___| \_|  |_/_|_| |_|\___|_|    v{}
                     | |
                     |_|                                                      {}"#,
        env!("CARGO_PKG_VERSION"),
        "\n"
    );

    let mut config = MinerConfig::load();

    if let Some(token_o) = args.token {
        config.token_address = token_o.parse().expect("Could not parse token address");
    }

    // Create GPU instance with optional index override from CLI
    let gpu = std::sync::Arc::new(Gpu::new(args.gpu)?);

    miner(gpu, &config).await;

    Ok(())
}
