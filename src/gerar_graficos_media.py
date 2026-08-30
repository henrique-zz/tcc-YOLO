import pandas as pd
import matplotlib.pyplot as plt
import os

def gerar_grafico_medio():
    print("📊 Lendo os resultados das 10 replicações do FLAME com Fechamento Morfológico...")
    
    # Caminho base das pastas 
    base_path = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train\flame-morph-200epochs"
    
    # Range de 1 a 11 para incluir do 1 ao 10
    pastas = [os.path.join(base_path, f"flame-morph-200epochs-yolov8n-run-{i}") for i in range(1, 2)]

    dfs = []
    for pasta in pastas:
        csv_path = os.path.join(pasta, "results.csv")
        try:
            df = pd.read_csv(csv_path)
            # O YOLO coloca espaços nos nomes das colunas, isso limpa tudo
            df.columns = df.columns.str.strip()
            dfs.append(df)
        except FileNotFoundError:
            print(f"⚠️ Erro: Não achei o arquivo na pasta {pasta}. Confirme o nome exato!")
            return

    print("🧮 Calculando a média das 200 épocas para as 10 rodadas...")
    # Soma todos os DataFrames da lista e divide pela quantidade exata lida (10)
    df_mean = sum(dfs) / len(dfs)

    print("🎨 Desenhando o gráfico...")
    plt.figure(figsize=(14, 5))

    # Gráfico 1: A Perda (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(df_mean['epoch'], df_mean['train/box_loss'], label='Erro da Caixa (Box Loss)', color='#d62728', linewidth=2)
    plt.plot(df_mean['epoch'], df_mean['train/cls_loss'], label='Erro da Classe (Class Loss)', color='#1f77b4', linewidth=2)
    plt.title('Média de Erro (Loss) - 10 Replicações FLAME')
    plt.xlabel('Épocas')
    plt.ylabel('Valor do Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: A Precisão (mAP)
    plt.subplot(1, 2, 2)
    plt.plot(df_mean['epoch'], df_mean['metrics/mAP50(B)'], label='mAP50', color='#2ca02c', linewidth=2)
    plt.title('Média de Desempenho - 10 Replicações FLAME')
    plt.xlabel('Épocas')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajusta o layout e salva a imagem na raiz do projeto
    plt.tight_layout()
    nome_saida = "grafico_media_flame_morph_200epochs_10runs.png"
    plt.savefig(nome_saida, dpi=300)
    print(f"✅ Sucesso! Gráfico salvo como: {nome_saida}")
    
    # Mostra o gráfico na tela
    plt.show()

if __name__ == "__main__":
    gerar_grafico_medio()