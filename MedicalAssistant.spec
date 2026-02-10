# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['desktop\\main.py'],
    pathex=['desktop'],
    binaries=[],
    datas=[('web_app/frontend/dist', 'dist'), ('saved_models/onnx', 'models'), ('saved_models/skin_disease_mapping.json', 'models')],
    hiddenimports=['inference', 'onnxruntime', 'cv2', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'torch', 'tensorboard', 'keras', 'matplotlib', 'ipython', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MedicalAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
