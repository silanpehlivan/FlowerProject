import os
import sys
import py_compile

ROOT = os.path.dirname(__file__)
required_files = [
    'app.py',
    'feature_extraction.py',
    'README.md',
    '.gitignore',
    'archive/README.txt'
]
# Optional but recommended model files
optional_models = [
    'tl_improved_svm_model.pkl',
    'tl_improved_scaler.pkl'
]

print('Validating project setup...')
errors = []
for f in required_files:
    path = os.path.join(ROOT, f)
    if not os.path.exists(path):
        errors.append(f'Missing required file: {f}')
    else:
        print(f'Found: {f}')

# Check optional models
for m in optional_models:
    path = os.path.join(ROOT, m)
    if not os.path.exists(path):
        print(f'Warning: Optional model not found: {m}')
    else:
        print(f'Optional model present: {m}')

# Syntax-check app.py
try:
    py_compile.compile(os.path.join(ROOT, 'app.py'), doraise=True)
    print('Syntax check passed for app.py')
except py_compile.PyCompileError as e:
    errors.append(f'Syntax error in app.py: {e}')

if errors:
    print('\nValidation FAILED with errors:')
    for e in errors:
        print(' -', e)
    sys.exit(2)

print('\nAll required checks passed.')
sys.exit(0)
