# StarNodes v2.0.0 - Installation Guide

## 📋 Prerequisites

- ComfyUI installed and working
- Python 3.10 or higher
- pip package manager
- Git (for git installation method)

---

## 🚀 Installation Methods

### Method 1: ComfyUI Manager (Recommended)

**Easiest method for most users**

1. Open ComfyUI
2. Click on "Manager" button
3. Click "Install Custom Nodes"
4. Search for "StarNodes"
5. Click "Install"
6. Restart ComfyUI

✅ **Advantages:**
- Automatic dependency installation
- Easy updates
- No command line needed

---

### Method 2: Git Clone (For Developers)

**Best for staying up-to-date with latest changes**

```bash
# Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes/

# Clone the repository
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes.git comfyui_starnodes

# Navigate into the directory
cd comfyui_starnodes

# Install dependencies
pip install -r requirements.txt

# Restart ComfyUI
```

✅ **Advantages:**
- Easy to update with `git pull`
- Can switch between versions
- Can contribute to development

---

### Method 3: Manual Installation (Release Package)

**For offline installations or specific version control**

1. **Download Release Package**
   - Download the `release2.0.0` folder
   - Or download from GitHub releases page

2. **Copy to ComfyUI**
   ```bash
   # Copy the folder to ComfyUI custom_nodes
   cp -r release2.0.0 /path/to/ComfyUI/custom_nodes/comfyui_starnodes
   ```

3. **Install Dependencies**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/comfyui_starnodes
   pip install -r requirements.txt
   ```

4. **Restart ComfyUI**

✅ **Advantages:**
- Works offline
- Version locked
- Clean installation

---

## 📦 Dependencies

StarNodes v2.0.0 requires the following Python packages:

```
requests>=2.31.0          # HTTP requests
beautifulsoup4>=4.12.0    # HTML parsing
newspaper3k>=0.2.8        # News scraping
lxml[html_clean]>=5.3.1   # XML/HTML processing
psd-tools>=1.10.0         # PSD file handling
opencv-python>=4.8.0      # Image processing
webcolors>=1.13.0         # Color utilities
color-matcher             # Color matching
soundfile>=0.12.0         # Audio file handling (NEW in v2.0.0)
```

### Installing Dependencies Manually

If automatic installation fails:

```bash
pip install requests>=2.31.0
pip install beautifulsoup4>=4.12.0
pip install newspaper3k>=0.2.8
pip install "lxml[html_clean]>=5.3.1"
pip install psd-tools>=1.10.0
pip install opencv-python>=4.8.0
pip install webcolors>=1.13.0
pip install color-matcher
pip install soundfile>=0.12.0
```

---

## 🎵 Optional: ACE Step Music Generator Setup

The **Star ACE Step Music Generator** requires a local ACE Step 1.5 API server.

### Setup ACE Step API

1. **Download ACE Step 1.5**
   - Get from official ACE Step repository
   - Follow their installation instructions

2. **Start API Server**
   ```bash
   # Start the ACE Step API server
   python ace_step_api.py --port 8000
   ```

3. **Configure in ComfyUI**
   - The node will connect to `http://localhost:8000` by default
   - You can change the endpoint in the node settings

4. **Test Connection**
   - Add the node to your workflow
   - Try generating a short music clip
   - Check the console for any errors

⚠️ **Note:** Music generation is optional. All other nodes work without ACE Step.

---

## 🎬 Optional: LTX Video Setup

The **LTX Video Toolz** nodes require LTX Video models.

### Setup LTX Video

1. **Download LTX Video Model**
   - Get from Hugging Face or official source
   - Place in ComfyUI models directory

2. **Download LTX VAE**
   - Required for encoding/decoding
   - Place in ComfyUI VAE directory

3. **Configure Paths**
   - ComfyUI will auto-detect models
   - Select in node dropdowns

⚠️ **Note:** LTX Video nodes are optional. All other nodes work without LTX models.

---

## ✅ Verify Installation

### Check Node Availability

1. **Start ComfyUI**
   ```bash
   python main.py
   ```

2. **Open in Browser**
   - Navigate to `http://localhost:8188`

3. **Add Node**
   - Right-click on canvas
   - Search for "Star"
   - You should see all 86 StarNodes

4. **Check Console**
   - Look for: `StarNodes: Successfully loaded`
   - Check for any error messages

### Test Basic Functionality

1. **Add a Simple Node**
   - Try `Star Text Input` or `Star Node`
   - Connect to workflow
   - Run generation

2. **Check Wildcards**
   - Wildcards should be copied to `ComfyUI/wildcards/`
   - Check console for copy confirmation

3. **Test Theme System**
   - Right-click on a StarNode
   - Look for theme options in menu
   - Try applying a theme

---

## 🔧 Troubleshooting

### Issue: Nodes Not Appearing

**Solution:**
```bash
# Reinstall dependencies
cd ComfyUI/custom_nodes/comfyui_starnodes
pip install -r requirements.txt --force-reinstall

# Restart ComfyUI
```

### Issue: Import Errors

**Solution:**
```bash
# Check Python version (must be 3.10+)
python --version

# Upgrade pip
pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Missing Wildcards

**Solution:**
- Wildcards are auto-copied on first load
- Check `ComfyUI/wildcards/` directory
- Manually copy from `comfyui_starnodes/wildcards/` if needed

### Issue: Music Generation Not Working

**Solution:**
- Ensure ACE Step API is running
- Check API endpoint in node settings
- Verify network connectivity
- Check console for error messages

### Issue: LTX Video Errors

**Solution:**
- Verify LTX models are installed
- Check model paths in ComfyUI
- Ensure sufficient VRAM
- Check console for specific errors

### Issue: Theme System Not Working

**Solution:**
- Restart ComfyUI after installing
- Check ComfyUI settings for theme options
- Verify JavaScript files loaded (check browser console)

---

## 🔄 Updating StarNodes

### From ComfyUI Manager

1. Open ComfyUI Manager
2. Click "Update All" or find StarNodes
3. Click "Update"
4. Restart ComfyUI

### From Git

```bash
cd ComfyUI/custom_nodes/comfyui_starnodes
git pull origin main
pip install -r requirements.txt --upgrade
# Restart ComfyUI
```

### Manual Update

1. Download new release
2. Backup your current installation
3. Replace files with new version
4. Reinstall dependencies
5. Restart ComfyUI

---

## 🗑️ Uninstalling

### Complete Removal

```bash
# Stop ComfyUI first

# Remove StarNodes directory
cd ComfyUI/custom_nodes/
rm -rf comfyui_starnodes

# Optional: Remove wildcards
rm -rf ../wildcards/

# Restart ComfyUI
```

### Keep Wildcards

```bash
# Only remove StarNodes, keep wildcards
cd ComfyUI/custom_nodes/
rm -rf comfyui_starnodes

# Restart ComfyUI
```

---

## 📞 Getting Help

### Documentation
- **README.md** - Main documentation
- **NODES_LIST_V2.md** - All nodes reference
- **RELEASE_SUMMARY_2.0.0.md** - Release details
- **CHANGELOG_2.0.0.md** - What's changed

### Support Channels
- **GitHub Issues:** Report bugs and request features
- **GitHub Discussions:** Ask questions and share workflows
- **ComfyUI Discord:** Community support

### Before Asking for Help

1. Check this installation guide
2. Read the error message carefully
3. Check GitHub issues for similar problems
4. Verify all dependencies are installed
5. Try reinstalling StarNodes

### Reporting Issues

When reporting issues, include:
- ComfyUI version
- StarNodes version (2.0.0)
- Python version
- Operating system
- Full error message
- Steps to reproduce
- Screenshots if relevant

---

## 🎉 You're Ready!

StarNodes v2.0.0 is now installed and ready to use.

**Next Steps:**
1. Explore the 86 available nodes
2. Check example workflows
3. Read node documentation
4. Join the community
5. Create amazing content!

**Happy Creating! 🌟**
