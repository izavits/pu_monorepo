#
# Create output directories to store results for each dataset
#

import os


def create_output_directories(parent, datasets, label, ratio, loss, method, prior, ad_step):
    for dataset in datasets:
        if prior:
            if ad_step:
                directory_name = f'{parent}/{dataset}_{label}_{ratio}_{ad_step}_{loss}_{method}_{prior}'
            else:
                directory_name = f'{parent}/{dataset}_{label}_{ratio}_{loss}_{method}_{prior}'
        else:
            if ad_step:
                directory_name = f'{parent}/{dataset}_{label}_{ratio}_{ad_step}_{loss}_{method}'
            else:
                directory_name = f'{parent}/{dataset}_{label}_{ratio}_{loss}_{method}'
        try:
            os.mkdir(directory_name)
            print(f'Directory {directory_name} created successfully.')
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    data = [
        'SCALED_CARDIO',
        'SCALED_CLIMATE',
        'SCALED_COVER',
        'SCALED_LETTER',
        'SCALED_MAMMOGRAPHY',
        'SCALED_PIECHART',
        'SCALED_PIZZACUTTER',
        'SCALED_POKER',
        'SCALED_SATELLITE',
        'SCALED_SEGMENT',
        'SCALED_SEISMIC',
        'SCALED_SHUTTLE',
        'SCALED_THYROID',
        'SCALED_YEAST'
    ]
    label_mechanism = 'SAR'
    ratio = 0.75
    loss = 'iFPNU'
    method = 'xgb'
    prior = None
    parent_dir = './output_2step_xgb'
    ad_step = 'dif'
    create_output_directories(parent_dir, data, label_mechanism, ratio, loss, method, prior, ad_step)