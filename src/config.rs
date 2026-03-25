use alloy::primitives::{address, Address};
use serde::{Deserialize, Serialize}; // Add Serialize
use std::{fs, path::Path};

#[derive(Deserialize, Serialize, Clone)] // Add Serialize here
pub struct MinerConfig {
    pub private_key: String,
    pub token_address: Address,
    pub batch_size: Option<u32>,
    pub refresh_interval: Option<u32>,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            private_key: "0x0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
            token_address: address!("0x0000000000000000000000000000000000000000"),
            batch_size: None,
            refresh_interval: Some(3),
        }
    }
}

impl MinerConfig {
    pub fn load() -> Self {
        let path = Path::new("config.toml");

        if !path.exists() {
            let default_config = Self::default();
            // Convert struct to TOML string
            let toml_str = toml::to_string_pretty(&default_config)
                .expect("Failed to serialize default config");

            // Write to file
            fs::write(path, toml_str).expect("Failed to create config.toml");

            println!("-------------------------------------------------------");
            println!("Empty config.toml created!");
            println!("Please edit config.toml with your PRIVATE_KEY and restart.");
            println!("-------------------------------------------------------");
            std::process::exit(0); // Exit so the user can edit the file
        }

        let config_str = fs::read_to_string(path).expect("Failed to read config.toml");

        toml::from_str(&config_str).expect("Failed to parse config.toml")
    }
}
