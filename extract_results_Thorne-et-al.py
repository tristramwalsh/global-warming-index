import os
import sys
import pandas as pd

scenarios = ['observed-SSP245',
             'SMILE_ESM-SSP126', 'SMILE_ESM-SSP245', 'SMILE_ESM-SSP370',
             'NorESM_rcp45-Volc', 'NorESM_rcp45-VolcConst']
variables = 'GHG-Nat-OHF'
ensembles = 'all'
headlines = ['ANNUAL', 'SR15', 'AR6', 'CGWL']

for scenario in scenarios:
    print(scenario)
    # Get a list of all available ensembles in th scenario directory
    ensembles_seletions_all = sorted(
        [d.split('ENSEMBLE-MEMBER--')[1] for d in
         os.listdir(f'results/aggregated/SCENARIO--{scenario}/')])
    for ensemble in ensembles_seletions_all:
        print('  ', ensemble)
        for headline in headlines:
            print('    ', headline)
            # Find the file in the results/aggregated directory
            base_dir = ('results/aggregated/' +
                        f'SCENARIO--{scenario}/' +
                        f'ENSEMBLE-MEMBER--{ensemble}/' +
                        f'VARIABLES--{variables}/'
                        # f'GWI_results_{headline}_HISTORICAL-ONLY/'
                        )

            hist_file = [f for f in os.listdir(base_dir)
                         if f.endswith('.csv') and headline in f
                         ][0]

            # Load the HISTORICAL_ONLY data
            hist_data = pd.read_csv(base_dir + hist_file,
                                    index_col=0,  header=[0, 1], skiprows=0)
            # Set pandas display options to show all columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            # print(hist_data.head())
            # Select only Nat, Ant, Tot:
            hist_data = hist_data.loc[:, ['Nat', 'Ant', 'Tot']]
            # print(hist_data.head())
            # Check and make the directory if it doesn't exist
            if not os.path.exists('Thorne2025'):
                os.makedirs('Thorne2025')
            # Save the data to the new directory
            hist_data.to_csv(f'Thorne2025/{hist_file}')

        # Copy the figure from plots/aggregated into the Thorne2025 directory
        # Check if the file exists before copying
        plot_dir = base_dir.replace('results', 'plots')
        fig = [f for f in os.listdir(plot_dir)
               if f.endswith('.png') and 'Historical_and_full_headlines' in f
               ][0]
        fig_file = plot_dir + fig
        os.system(f'cp {fig_file} Thorne2025/{fig}')
