# Star Divisible Dimension

## Description
This node ensures that image dimensions (width and height) are divisible by a specified value. This is particularly useful for ensuring compatibility with VAE models and other components that require dimensions to be divisible by specific values (commonly 8, 16, or 32).

## Inputs
- **width**: The input width to adjust (default: 512, range: 64-8192)
- **height**: The input height to adjust (default: 512, range: 64-8192)
- **divisible_by**: The value that both dimensions should be divisible by (default: 8, range: 1-64)

## Outputs
- **width**: The adjusted width that is divisible by the specified value
- **height**: The adjusted height that is divisible by the specified value

## Usage
1. Input your desired width and height
2. Specify the value that dimensions should be divisible by (commonly 8 for most VAEs)
3. The node will round both dimensions to the nearest multiple of the divisible_by value
4. Use the adjusted dimensions in your image generation workflow

This node is particularly useful for:
- Ensuring VAE compatibility by making dimensions divisible by 8
- Preparing dimensions for specific models that require dimensions divisible by 16 or 32
- Avoiding errors in image generation due to incompatible dimensions
- Standardizing dimensions across workflows
