import pandas as pd
import os
import numpy as np

def extrair_metricas_finais(dataset_nome, caminho_base, prefixo_run, total_runs=10):
    print(f"\n=======================================================")
    print(f"📊 EXTRAINDO RESULTADOS: {dataset_nome.upper()}")
    print(f"=======================================================\n")

    melhores_maps = []
    melhores_maps_95 = []
    melhores_precisions = []
    melhores_recalls = []

    for i in range(1, total_runs + 1):
        pasta_run = os.path.join(caminho_base, f"{prefixo_run}{i}")
        caminho_csv = os.path.join(pasta_run, "results.csv")

        try:
            df = pd.read_csv(caminho_csv)
            df.columns = df.columns.str.strip()

            linha_best = df.loc[df['metrics/mAP50(B)'].idxmax()]

            best_map = linha_best['metrics/mAP50(B)']
            best_map_95 = linha_best['metrics/mAP50-95(B)']
            best_precision = linha_best['metrics/precision(B)']
            best_recall = linha_best['metrics/recall(B)']
            best_epoch = linha_best['epoch']

            melhores_maps.append(best_map)
            melhores_maps_95.append(best_map_95)
            melhores_precisions.append(best_precision)
            melhores_recalls.append(best_recall)

        except FileNotFoundError:
            pass
        except KeyError as e:
            print(f"⚠️ Erro nas colunas: {e}")

    if len(melhores_maps) > 0:
        print(f"📈 --- ESTATÍSTICAS FINAIS PARA A TABELA DO OVERLEAF ({len(melhores_maps)} Runs) ---")
        print(f"Precision  -> Média: {np.mean(melhores_precisions):.4f} | Máxima: {np.max(melhores_precisions):.4f} | Mínima: {np.min(melhores_precisions):.4f}")
        print(f"Recall     -> Média: {np.mean(melhores_recalls):.4f} | Máxima: {np.max(melhores_recalls):.4f} | Mínima: {np.min(melhores_recalls):.4f}")
        print(f"mAP@50     -> Média: {np.mean(melhores_maps):.4f} | Máxima: {np.max(melhores_maps):.4f} | Mínima: {np.min(melhores_maps):.4f}")
        print(f"mAP@50-95  -> Média: {np.mean(melhores_maps_95):.4f} | Máxima: {np.max(melhores_maps_95):.4f} | Mínima: {np.min(melhores_maps_95):.4f}")
        print("-------------------------------------------------------\n")

if __name__ == "__main__":
    # Caminhos configurados para o dataset FlameVision
    base_flamevision = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train\corsican-morph-200epochs"
    prefixo_flamevision = "corsican-morph-200epochs-yolov8n-run-"

    extrair_metricas_finais("CORSICAN", base_flamevision, prefixo_flamevision, total_runs=10)