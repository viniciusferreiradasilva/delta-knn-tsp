#!/usr/bin/env bash
# Exemplo de execução de uma predição.
#python atividade04/prediction.py --input 'input/all_stocks_5yr.csv' --stock_name 'NVDA' --field 'open' --window 10 --steps 3 --regressor 2

# Exemplo de execução do ajuste de parâmetros.
python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'NVDA' --field 'open'  --from_window 10 --to_window 150 --steps 1 3 5 --configs '0' '2 1' '3 150' '4'
python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'CHK' --field 'open'  --from_window 10 --to_window 150 --steps 1 3 5 --configs '0' '2 1' '3 150' '4'
python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'AES' --field 'open'  --from_window 10 --to_window 150 --steps 1 3 5 --configs '0' '2 1' '3 150' '4'
python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'PCLN' --field 'open'  --from_window 10 --to_window 150 --steps 1 3 5 --configs '0' '2 1' '3 150' '4'

#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'NVDA' --field 'open'  --from_window 10 --to_window 150 --from_parameter 1 --to_parameter 1 --steps 1 3 5 --regressor 2
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'CHK' --field 'open'  --from_window 10 --to_window 150 --from_parameter 1 --to_parameter 1 --steps 1 3 5 --regressor 2
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'AES' --field 'open'  --from_window 10 --to_window 150 --from_parameter 1 --to_parameter 1 --steps 1 3 5 --regressor 2
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'PCLN' --field 'open'  --from_window 10 --to_window 150 --from_parameter 1 --to_parameter 1 --steps 1 3 5 --regressor 2

#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'NVDA' --field 'open'  --from_window 10 --to_window 150 --from_parameter 150 --to_parameter 150 --steps 1 3 5 --regressor 4
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'CHK' --field 'open'  --from_window 10 --to_window 150 --from_parameter 150 --to_parameter 150 --steps 1 3 5 --regressor 4
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'AES' --field 'open'  --from_window 10 --to_window 150 --from_parameter 150 --to_parameter 150 --steps 1 3 5 --regressor 4
#python atividade04/parameter_fitting.py --input 'input/all_stocks_5yr.csv' --stock_name 'PCLN' --field 'open'  --from_window 10 --to_window 150 --from_parameter 150 --to_parameter 150 --steps 1 3 5 --regressor 4