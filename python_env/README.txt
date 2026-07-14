DeOldify Python Environment for JHenTai (Windows x64)
======================================================

This package contains an embeddable Python 3.11.9 environment with all
Python dependencies required by the DeOldify colorization feature:
  - onnxruntime
  - numpy
  - Pillow
  - flatbuffers, packaging, protobuf (transitive deps)

IMPORTANT - Installation Directory:
  All dependencies are installed INSIDE the Python extraction directory
  at: <extracted folder>\python\Lib\site-packages\
  NOTHING is written to C:\Users\<yourname>\ or the system registry.

How to use
----------
1. Unzip this archive anywhere on your Windows PC, e.g.
      D:\Tools\deoldify_env\

2. Double-click "install.bat" and wait for the installation to finish.
   It will install pip and the offline wheels into:
      D:\Tools\deoldify_env\python\Lib\site-packages\

3. After installation, open JHenTai -> Settings -> Advanced -> Colorization.
   Set the "Python Path" to:
      D:\Tools\deoldify_env\python\python.exe
   (adjust the drive/path to match where you extracted the archive)

4. Download the DeOldify ONNX model file and set the "Model Directory".

5. Restart JHenTai if it was already running, then use the colorize button.

Project source files included
-----------------------------
The "project_files" folder contains the modified Dart/Python source files
for the DeOldify feature. These are for reference; they are already present
in your checked-out branch (feature/AiColoring-add-deoldify).

Notes
-----
- This package is for Windows 64-bit only.
- No internet connection is required during "install.bat" because all
  wheels are bundled in the "wheels" folder.
- This package is NOT pushed to GitHub.
- To completely remove: just delete the deoldify_env folder.
