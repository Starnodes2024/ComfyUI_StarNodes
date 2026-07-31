# StarNodes 2.1.0 - Installation Guide

## Requirements
- ComfyUI (latest version recommended)
- Python 3.10 or higher
- Git

## Installation Methods

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI
2. Click on "Manager" button
3. Search for "StarNodes"
4. Click "Install"
5. Restart ComfyUI

### Method 2: Manual Installation via Git
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes.git
cd ComfyUI_StarNodes
pip install -r requirements.txt
```

### Method 3: Manual Installation from Release Package
1. Download the release2.1.0 folder
2. Extract to `ComfyUI/custom_nodes/ComfyUI_StarNodes`
3. Open terminal in the extracted folder
4. Run: `pip install -r requirements.txt`
5. Restart ComfyUI

## Updating from Previous Version

### Via ComfyUI Manager
1. Open ComfyUI Manager
2. Go to "Update" tab
3. Find "StarNodes" and click "Update"
4. Restart ComfyUI

### Via Git
```bash
cd ComfyUI/custom_nodes/ComfyUI_StarNodes
git pull
pip install -r requirements.txt --upgrade
```

## Verifying Installation

After installation and restarting ComfyUI:
1. Right-click in the ComfyUI canvas
2. Navigate to "Add Node" → "⭐StarNodes"
3. You should see all 88 StarNodes organized in categories:
   - Image And Latent
   - Text And Data
   - Helpers And Tools
   - Sampling
   - External
   - Grid
   - Music
   - Qwen
   - LTX Video

## Dependencies

StarNodes automatically installs the following dependencies:
- `soundfile>=0.12.0` (for music generation nodes)
- Standard Python libraries (PIL, numpy, cv2, etc.)

### Optional Dependencies
Some nodes may require additional dependencies:
- **Music Generation Nodes**: Require ACE Step 1.5 API running locally
- **Qwen Nodes**: Require Qwen models installed in ComfyUI

## Troubleshooting

### Issue: Nodes not appearing
**Solution:** 
- Ensure you restarted ComfyUI after installation
- Check the console for error messages
- Verify the installation path is correct

### Issue: Import errors
**Solution:**
```bash
cd ComfyUI/custom_nodes/ComfyUI_StarNodes
pip install -r requirements.txt --force-reinstall
```

### Issue: Music generation not working
**Solution:**
- Ensure ACE Step 1.5 API is running on `http://localhost:8001`
- Check the API endpoint in the node settings

### Issue: Wildcards not working
**Solution:**
- StarNodes automatically copies wildcards to ComfyUI's main directory
- If issues persist, manually copy the `wildcards` folder to `ComfyUI/wildcards`

## Uninstallation

### Via ComfyUI Manager
1. Open ComfyUI Manager
2. Find "StarNodes" in the installed list
3. Click "Uninstall"
4. Restart ComfyUI

### Manual Uninstallation
```bash
cd ComfyUI/custom_nodes
rm -rf ComfyUI_StarNodes  # Linux/Mac
# or
rmdir /s ComfyUI_StarNodes  # Windows
```

## Support

- **GitHub Issues**: https://github.com/Starnodes2024/ComfyUI_StarNodes/issues
- **Documentation**: See `web/docs` folder for individual node help files
- **Changelog**: See `CHANGELOG.md` for version history

## What's New in 2.1.0

- **New Nodes**: Star Box Drawer, Star Image Shifter
- **Enhancement**: Star Save Panorama JPG+ now outputs the 3D image
- **Documentation**: Comprehensive help files for all new features

For detailed changes, see `RELEASE_NOTES_2.1.0.md` and `CHANGELOG.md`.
