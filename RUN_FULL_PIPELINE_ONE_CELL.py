# Paste this entire file into one Google Colab cell after uploading/extracting the project ZIP.
from google.colab import drive, files
from pathlib import Path
import os, sys, subprocess, shutil

drive.mount('/content/drive')

PROJECT = Path('/content/drive/MyDrive/Paper2_MOF_Screening_FULL_CORRECTED')
if not PROJECT.exists():
    print('Upload Paper2_MOF_Screening_FULL_CORRECTED_CODE.zip')
    uploaded = files.upload()
    zips = [Path(name) for name in uploaded if name.lower().endswith('.zip')]
    if not zips:
        raise FileNotFoundError('No ZIP uploaded.')
    import zipfile
    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0], 'r') as zf:
        zf.extractall(PROJECT.parent)
    # Support either root folder name.
    candidates = [PROJECT] + [p for p in PROJECT.parent.iterdir() if p.is_dir() and (p/'src'/'run_pipeline.py').exists()]
    PROJECT = next((p for p in candidates if (p/'src'/'run_pipeline.py').exists()), None)
    if PROJECT is None:
        raise FileNotFoundError('Valid project root not found after extraction.')

os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

subprocess.run([sys.executable, 'tests/test_model_smoke.py'], check=True)
subprocess.run([sys.executable, '-u', 'src/run_pipeline.py'], check=True)
subprocess.run([sys.executable, 'src/run_checks.py'], check=True)

zip_base = '/content/Paper2_MOF_Screening_FINAL_RESULTS'
zip_path = shutil.make_archive(zip_base, 'zip', root_dir=PROJECT, base_dir='results')
drive_zip = Path('/content/drive/MyDrive/Paper2_MOF_Screening_FINAL_RESULTS.zip')
shutil.copy2(zip_path, drive_zip)
print('Final Drive ZIP:', drive_zip)
files.download(zip_path)
