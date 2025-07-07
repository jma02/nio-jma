import os
import fileinput

def rename_dataset_class(file_path, old_name, new_name):
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            print(line.replace(old_name, new_name), end='')

# Define the mapping of old to new names
dataset_renames = {
    'Helm32L.py': 'Helmholtz32L',
    'HelmMoreInc.py': 'HelmholtzMoreInc',
    'HelmRandom.py': 'HelmholtzRandom',
    'HeartLungsOS.py': 'HeartLungsOS',
    'HeartLungsEIT.py': 'HeartLungsEIT',
    'PoissonSin.py': 'PoissonSin',
    'PoissonSin200L.py': 'PoissonSin200L',
    'HelmNIO.py': 'HelmholtzNIO',
    'HelmholtzGRF.py': 'HelmholtzGRF',
    'AlbedoOperator.py': 'AlbedoOperator',
    'CurveVel.py': 'CurveVel',
    'StlyleData.py': 'StyleData'
}

# Get the path to the Problems directory
problems_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Problems')

# Rename dataset classes in each problem file
for file_name, new_name in dataset_renames.items():
    file_path = os.path.join(problems_dir, file_name)
    if os.path.exists(file_path):
        rename_dataset_class(file_path, 'Helmholtz32LDataset', f'{new_name}Dataset')

# Update references in other files
for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    for file in files:
        if file.endswith('.py') and 'Problems' not in root:
            file_path = os.path.join(root, file)
            for old_name, new_name in dataset_renames.items():
                # Update imports
                rename_dataset_class(file_path, f'from Problems.{old_name} import Helmholtz32LDataset', 
                                    f'from Problems.{old_name} import {new_name}Dataset')
                # Update dataset instances
                rename_dataset_class(file_path, 'Helmholtz32LDataset', f'{new_name}Dataset')
