# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# GCC library paths
gcc_lib_path = "/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/14"

# OpenBLAS paths
openblas_lib_path = "/opt/homebrew/opt/openblas/lib"

# Get all PaddleOCR submodules
paddle_submodules = collect_submodules('paddleocr')
paddle_datas = collect_data_files('paddleocr', include_py_files=True)

# Prepare binary list with actual files that exist
binary_list = [
    # GCC libraries
    (f"{gcc_lib_path}/libgfortran.5.dylib", '.'),
    (f"{gcc_lib_path}/libquadmath.0.dylib", '.'),
    (f"{gcc_lib_path}/libgcc_s.1.1.dylib", '.'),
    # OpenBLAS libraries
    (f"{openblas_lib_path}/libopenblas.0.dylib", '.'),
    (f"{openblas_lib_path}/libopenblasp-r0.3.28.dylib", '.')
]

a = Analysis(
    ['flashcard_gui.py'],
    pathex=[],
    binaries=binary_list,
    datas=paddle_datas + collect_data_files('paddle', include_py_files=True) + 
          collect_data_files('Cython') + collect_data_files('numpy'),
    hiddenimports=[
        'paddleocr', 'paddle', 'easyocr', 'Cython', 'numpy',
        'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'paddleocr.tools', 'paddleocr.ppocr', 'paddleocr.ppstructure',
        'paddleocr.tools.infer', 'paddleocr.tools.eval',
        'paddleocr.tools.export_model', 'paddleocr.ppocr.utils',
        'Cython.Compiler.Main', 'Cython.Compiler.Pipeline',
        'Cython.Compiler.Visitor', 'Cython.Compiler.PyrexTypes',
        'Cython.Compiler.Parsing', 'Cython.Runtime.refnanny'
    ] + paddle_submodules + collect_submodules('paddle'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Add explicit PaddleOCR tools module
import site
site_packages = site.getsitepackages()[0]
paddleocr_tools = os.path.join(site_packages, 'paddleocr', 'tools')
if os.path.exists(paddleocr_tools):
    a.datas += [(f'paddleocr/tools/{f}', os.path.join(paddleocr_tools, f), 'DATA')
                for f in os.listdir(paddleocr_tools) if f.endswith('.py')]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='flashcard_gui',
    debug=True,  # Keep debug mode on for now
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Turn to true for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='flashcard_gui.app',
    icon=None,
    bundle_identifier=None,
)