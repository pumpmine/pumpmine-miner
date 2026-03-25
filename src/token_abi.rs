use alloy::sol;

sol! {
    #[sol(rpc)]
    #[derive(Debug)]
    interface PumpmineToken {
        function getMiningInfo()
            external
            view
            returns (
                bytes32 _challenge,
                uint256 _difficulty,
                uint256 _miningTarget,
                uint256 _totalMined,
                uint256 _mineableSupply,
                uint256 _emissionEndBlock,
                uint256 _blocksRemaining,
                uint256 _currentReward
            );
        function mine(uint256 nonce) external nonReentrant;

        // --- Standard ERC20 Metadata ---
        function name() external view returns (string memory);
        function symbol() external view returns (string memory);
        function decimals() external view returns (uint8);

        // --- Standard ERC20 Logic ---
        function totalSupply() external view returns (uint256);
        function balanceOf(address account) external view returns (uint256);
        function transfer(address to, uint256 amount) external returns (bool);
    }
}
