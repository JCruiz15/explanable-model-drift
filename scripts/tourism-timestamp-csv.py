import os
import pandas as pd
from argparse import ArgumentParser

def tourism_excel_to_csv(
        path: str,
        sheet_name: str,
        output: str,
        skiprows: int = 10
    ):
    # Load the data
    df = pd.read_excel(path, engine='openpyxl', sheet_name=sheet_name, skiprows=skiprows)

    # PREPROCESSING
    df.replace('-', None, inplace=True)
    df.replace(':', None, inplace=True)
 
    df['Búsquedas hacia AGP total'].fillna(0, inplace=True)
    df['Búsquedas hacia AGP  3 meses'].fillna(0, inplace=True)
    df['Búsquedas hacia AGP 6 meses'].fillna(0, inplace=True)

    df['Asientos ofertados'].fillna(-1, inplace=True)
    df.ffill(inplace=True)
    
    # Year-month into timestamp
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
    df.drop(columns=['Año', 'Mes'], inplace=True)
    
    df.to_csv(output, index=False, sep=';', encoding='utf-8', date_format='%Y-%m', decimal='.')


# 'resources/turism/overnights/Fichero_para_Predicciones_Turisticas_Pernoctaciones4.xlsx'
# 'Suecia'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--sheet-name', type=str, required=True)
    parser.add_argument('--skiprows', type=int, default=10)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    tourism_excel_to_csv(
        path=args.path,
        sheet_name=args.sheet_name,
        skiprows=args.skiprows,
        output=args.output
    )