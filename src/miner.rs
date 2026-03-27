use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
    u32,
};

use alloy::{
    hex,
    network::EthereumWallet,
    primitives::{utils::format_units, U256},
    providers::ProviderBuilder,
    signers::local::PrivateKeySigner,
};

use crate::{config::MinerConfig, cpu_keccak256, mine, token_abi::PumpmineToken, Gpu, Push};

// ── Batch size estimator ──────────────────────────────────────────────────────

pub fn estimate_safe_batch(gpu: &Gpu) -> u32 {
    // 500k is usually the minimum to bypass driver noise.
    let test_batch = 500_000u32;

    let dummy_push = Push {
        ch: [0x12345678u32; 8],
        sn: [0xABCDEF01u32; 5],
        base_lo: 0x98765432, // Non-zero!
        base_hi: 0x12345678, // Non-zero!
        tg: [0u32; 8],
    };

    // Warm up the GPU (Intel drivers need this to ramp up clocks)
    let _ = gpu.dispatch(&dummy_push, test_batch / 256);

    let start = std::time::Instant::now();
    // Run a slightly larger batch for the actual test
    let _ = gpu.dispatch(&dummy_push, test_batch / 256);
    let elapsed = start.elapsed().as_secs_f64();

    let target_time = 0.1f64; // 100ms

    // 3. Simple scaling is now more accurate because execution time > overhead
    let scale = target_time / elapsed;
    let raw_batch = (test_batch as f64 * scale).round() as u32;

    // Aligned to workgroup size (256)
    let batch = (raw_batch / 256) * 256;

    // Intel Arc A350M can easily handle 2 million+ per batch
    let safe_batch = batch.clamp(256, 4_000_000);

    println!(
        "[ESTIMATION] Real work test: {:.3} ms => Suggested batch: {}",
        elapsed * 1000.0,
        safe_batch
    );

    safe_batch
}

// ── Hashrate formatting ───────────────────────────────────────────────────────

fn format_hashrate(hashrate: f64) -> String {
    let units = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s"];
    let mut hr = hashrate;
    let mut unit_index = 0;
    while hr >= 1000.0 && unit_index < units.len() - 1 {
        hr /= 1000.0;
        unit_index += 1;
    }
    format!("{:.2} {}", hr, units[unit_index])
}

// ── Job snapshot shared between threads ──────────────────────────────────────

#[derive(Clone)]
struct Job {
    challenge: [u8; 32],
    target: [u8; 32],
    current_reward: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn miner(gpu: Arc<Gpu>, config: &MinerConfig) {
    // ── Batch size ────────────────────────────────────────────────────────────
    let recommended = estimate_safe_batch(&gpu);
    if let Some(user) = config.batch_size {
        if user > recommended {
            println!(
                "[WARNING] Batch size {user} exceeds recommended {recommended}. \
                 This may cause TDR crashes."
            );
        }
    }
    let batch_size = config.batch_size.unwrap_or(recommended);
    println!("[GPU] Using batch size {batch_size}");

    // ── Provider / contract ───────────────────────────────────────────────────
    let rpc = "https://mainnet.base.org".parse().unwrap();
    let signer: PrivateKeySigner = config.private_key.parse().expect("Invalid private key");
    let provider = Arc::new(
        ProviderBuilder::new()
            .wallet(EthereumWallet::from(signer.clone()))
            .connect_http(rpc),
    );
    let token = PumpmineToken::new(config.token_address, &provider);

    let token_symbol = token
        .symbol()
        .call()
        .await
        .expect("Failed to load token symbol");
    let token_name = token
        .name()
        .call()
        .await
        .expect("Failed to load token name");
    println!("\n[INFO] Loaded token {token_name} (${token_symbol})");

    // ── Initial job ───────────────────────────────────────────────────────────
    let info = token
        .getMiningInfo()
        .call()
        .await
        .expect("Could not fetch initial mining info");

    println!(
        "[INFO]  Current reward:    {} {token_symbol}\n\
         [INFO]  Total mined so far: {} {token_symbol}\n\
         [INFO]  Mining difficulty:  {}\n",
        format_units(info._currentReward, 18).unwrap(),
        format_units(info._totalMined, 18).unwrap(),
        info._difficulty,
    );
    
    if info._blocksRemaining <= 0 {
        eprintln!("Emission for {token_name} has ended.");
        return;
    }

    let initial_job = Job {
        challenge: info._challenge.0,
        target: info._miningTarget.to_be_bytes(),
        current_reward: format_units(info._currentReward, 18).unwrap(),
    };

    // ── Shared primitives ─────────────────────────────────────────────────────
    // The hashing thread reads the active job; the refresh task writes it.
    let shared_job: Arc<tokio::sync::RwLock<Job>> = Arc::new(tokio::sync::RwLock::new(initial_job));

    // Set to true by the refresh task when the challenge changes.
    let new_job_flag = Arc::new(AtomicBool::new(false));

    // Incremented by the hashing thread; read by the reporter.
    let hashes_done = Arc::new(AtomicU64::new(0));

    // Set to true while a solution is being submitted; hashing thread idles.
    let solution_pending = Arc::new(AtomicBool::new(false));

    // ── Hashrate reporter (tokio task) ────────────────────────────────────────
    {
        let hashes_done = hashes_done.clone();
        let interval = config.refresh_interval.unwrap_or(5) as u64;
        tokio::spawn(async move {
            let mut prev: u64 = 0;
            let mut tick = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                let now = hashes_done.load(Ordering::Relaxed);
                let secs = tick.elapsed().as_secs_f64();
                println!(
                    "[MINER] Hashrate: {}",
                    format_hashrate((now.saturating_sub(prev)) as f64 / secs)
                );
                prev = now;
                tick = Instant::now();
            }
        });
    }

    // ── Job refresh task (tokio task) ─────────────────────────────────────────
    // Polls the chain in the background; does NOT block the hashing thread.
    {
        let shared_job = shared_job.clone();
        let new_job_flag = new_job_flag.clone();
        let token_sym = token_symbol.clone();
        let token2 = PumpmineToken::new(config.token_address, provider.clone());
        let interval = config.refresh_interval.unwrap_or(10) as u64;

        tokio::spawn(async move {
            let mut last_challenge = { shared_job.read().await.challenge };

            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;

                let info = match token2.getMiningInfo().call().await {
                    Ok(i) => i,
                    Err(e) => {
                        eprintln!("[JOB] Fetch failed: {e}");
                        continue;
                    }
                };

                if info._challenge != last_challenge {
                    last_challenge = info._challenge.0;
                    let job = Job {
                        challenge: info._challenge.0,
                        target: info._miningTarget.to_be_bytes(),
                        current_reward: format_units(info._currentReward, 18).unwrap(),
                    };
                    println!(
                        "[JOB] New challenge: {}  diff={}  reward={} {token_sym}",
                        hex::encode(info._challenge),
                        info._difficulty,
                        job.current_reward,
                    );
                    *shared_job.write().await = job;
                    new_job_flag.store(true, Ordering::Release);
                }
            }
        });
    }

    // ── Solution submission channel ───────────────────────────────────────────
    // The blocking hashing thread sends solutions here; the async loop submits.
    let (sol_tx, mut sol_rx) = tokio::sync::mpsc::unbounded_channel::<(u64, [u8; 32], Job)>();

    // ── GPU hashing thread (std::thread so blocking dispatches don't starve tokio)
    {
        let gpu = gpu.clone();
        let shared_job = shared_job.clone();
        let new_job_flag = new_job_flag.clone();
        let hashes_done = hashes_done.clone();
        let solution_pending = solution_pending.clone();
        let signer_addr = *signer.address();

        std::thread::spawn(move || {
            // Grab initial job snapshot without async overhead.
            let mut active: Job = shared_job.blocking_read().clone();
            let mut base: u64 = 0;

            loop {
                // ── Pick up new job if the refresh task flagged one ───────────
                if new_job_flag
                    .compare_exchange(true, false, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    active = shared_job.blocking_read().clone();
                    base = 0;
                    println!("[MINER] Switched to new job, nonce search restarted.");
                }

                // ── Pause while a solution is being submitted ─────────────────
                if solution_pending.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }

                // ── One GPU batch ─────────────────────────────────────────────
                match mine(
                    &gpu,
                    &active.challenge,
                    &signer_addr,
                    &active.target,
                    base,
                    batch_size,
                ) {
                    Ok(Some((nonce, digest))) => {
                        println!(
                            "\n[SOLUTION] Found nonce={nonce} digest={}",
                            hex::encode(digest)
                        );
                        solution_pending.store(true, Ordering::Release);
                        let _ = sol_tx.send((nonce, digest, active.clone()));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[GPU] Dispatch error: {e}");
                        std::thread::sleep(Duration::from_secs(5));
                    }
                }

                hashes_done.fetch_add(batch_size as u64, Ordering::Relaxed);
                base = base.saturating_add(batch_size as u64);
            }
        });
    }

    // ── Solution submission loop (async, main task) ───────────────────────────
    let signer_addr = *signer.address();

    while let Some((nonce, digest, job)) = sol_rx.recv().await {
        // CPU-verify the solution before paying gas
        let expected = cpu_keccak256(&job.challenge, &signer_addr, nonce);
        if expected != digest {
            eprintln!("[SOLUTION] CPU verification FAILED (nonce={nonce}) - discarding.");
            solution_pending.store(false, Ordering::Release);
            continue;
        }

        // Estimate gas with a 1.5× safety margin
        let gas_limit = match token.mine(U256::from(nonce)).estimate_gas().await {
            Ok(g) => (g as f64 * 1.5) as u64,
            Err(e) => {
                eprintln!("[SOLUTION] Gas estimation failed ({e}), falling back to 200 000");
                200_000
            }
        };
        println!("[SOLUTION] Submitting with gas-limit={gas_limit}");

        match token.mine(U256::from(nonce)).gas(gas_limit).send().await {
            Err(e) => eprintln!("[SOLUTION] Send error: {e}"),
            Ok(pending) => match pending.get_receipt().await {
                Err(e) => eprintln!("[SOLUTION] Receipt error: {e}"),
                Ok(receipt) => {
                    if receipt.status() {
                        println!(
                            "[SOLUTION] Accepted!  Reward: {} {token_symbol}\n[SOLUTION]   block #{}\n[SOLUTION]   gas used: {}",
                            job.current_reward, receipt.block_number.unwrap_or_default(), receipt.gas_used,
                        );
                    } else {
                        eprintln!(
                            "[SOLUTION] Transaction reverted (stale nonce or insufficient gas)"
                        );
                    }
                }
            },
        };

        // Unblock the hashing thread regardless of outcome
        solution_pending.store(false, Ordering::Release);
    }
}
