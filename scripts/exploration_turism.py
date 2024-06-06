import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl as px

# Load the data
sheets = px.load_workbook('resources/turism/overnights/Fichero_para_Predicciones_Turisticas_Pernoctaciones4.xlsx').sheetnames
sheets = [sheet for sheet in sheets if sheet not in ['Notas Variables Explicativas', 'Total', 'Otros', 'Viajeros Hoteles y Apartamet', 'transformación']]

pernoctaciones = []
for sheet in sheets:
    df = pd.read_excel('resources/turism/overnights/Fichero_para_Predicciones_Turisticas_Pernoctaciones4.xlsx', engine='openpyxl', sheet_name=sheet, skiprows=10)
    df['Fecha'] = df['Año'].astype(str) + '-' + df['Mes'].astype(str)
    pernoctaciones.append(df)

for i in range (0, len(pernoctaciones)):
    # Plot the data
    plt.figure(figsize=(50, 20))
    sns.lineplot(x='Fecha', y=sheets[i], data=pernoctaciones[i])
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f'Pernoctaciones en {sheets[i]}')
    plt.savefig(f'results/exploration/tourism/overnights/overnights_{sheets[i]}.png', dpi=200, bbox_inches='tight')
    plt.close()