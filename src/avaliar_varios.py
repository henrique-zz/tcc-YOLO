import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

def avaliar_todas_rodadas():
    print("🔥 Iniciando Avaliação em Lote (Rodadas 1 a 10) 🔥")
    
    # --- CONFIGURAÇÃO DO DATASET ---
    dataset_nome = "corsican" 
    epochs = 150  
    
    # Dicionário para guardar os resultados de todas as rodadas
    acumulador_metricas = {
        "map50": [],
        "map50_95": [],
        "precision": [],
        "recall": []
    }
    
    relatorio_completo_terminal = ""

    # 1. Lê o arquivo de configurações para saber onde está o dataset
    caminho_config = Path("configs/config.yaml")
    if not caminho_config.exists():
        print("Erro: Arquivo config.yaml não encontrado.")
        return

    with open(caminho_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(".")
    pasta_dataset = base_dir / config['dataset']['pasta_saida']
    pasta_teste = pasta_dataset / "images" / "test"
    arquivo_yaml = pasta_dataset / "dataset.yaml"
    
    # --- A MÁGICA DA ORGANIZAÇÃO AQUI ---
    # Cria a "pasta-mãe" dentro de predict seguindo o mesmo padrão do train
    pasta_predict_mae = base_dir / "runs" / "predict" / f"{dataset_nome}-{epochs}epochs"

    # 2. O LOOP MÁGICO: Vai da rodada 1 até a 10
    for i in range(1, 11):
        nome_experimento = f"{dataset_nome}-{epochs}epochs-yolov8n-run{i}"
        
        pasta_treino = base_dir / "runs" / "train" / f"{dataset_nome}-{epochs}epochs" / nome_experimento
        caminho_pesos = pasta_treino / "weights" / "best.pt"
        
        print(f"\n{'=' * 50}")
        print(f"🚀 AVALIANDO: {nome_experimento}")
        print(f"{'=' * 50}")

        if not caminho_pesos.exists():
            texto_falha = f"⚠️ Arquivo 'best.pt' não encontrado para {nome_experimento}. Pulando...\n"
            print(texto_falha)
            relatorio_completo_terminal += texto_falha
            continue

        model = YOLO(caminho_pesos.resolve().as_posix())

        # Predição apontando para a pasta-mãe
        model.predict(
            source=pasta_teste.resolve().as_posix(),
            save=True,
            save_txt=True,
            save_conf=True,
            name=nome_experimento,
            project=pasta_predict_mae.resolve().as_posix(), # <--- Mudamos aqui!
        )

        # Arrumando a bagunça do YOLO dentro da pasta-mãe
        pasta_resultados = pasta_predict_mae / nome_experimento
        pasta_imagens_preditas = pasta_resultados / "images"
        pasta_imagens_preditas.mkdir(parents=True, exist_ok=True)

        for arquivo in pasta_resultados.glob("*.*"):
            if arquivo.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.move(str(arquivo), str(pasta_imagens_preditas / arquivo.name))

        metrics = model.val(
            data=arquivo_yaml.resolve().as_posix(),
            split="test",
        )
        
        acumulador_metricas["map50"].append(metrics.box.map50)
        acumulador_metricas["map50_95"].append(metrics.box.map)
        acumulador_metricas["precision"].append(metrics.box.mp)
        acumulador_metricas["recall"].append(metrics.box.mr)

        texto_metricas = (
            f"{'=' * 40}\n"
            f"MÉTRICAS: {nome_experimento}\n"
            f"{'=' * 40}\n"
            f"  mAP50     : {metrics.box.map50:.4f}\n"
            f"  mAP50-95  : {metrics.box.map:.4f}\n"
            f"  Precision : {metrics.box.mp:.4f}\n"
            f"  Recall    : {metrics.box.mr:.4f}\n"
            f"{'=' * 40}\n"
        )
        print(texto_metricas)
        relatorio_completo_terminal += texto_metricas + "\n"

        caminho_salvar = pasta_resultados / "metricas" / "metricas_finais.txt"
        caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_salvar, "w", encoding="utf-8") as f:
            f.write(texto_metricas)
            
    # 3. CÁLCULO E SALVAMENTO DA MÉDIA FINAL
    qtd_rodadas_sucesso = len(acumulador_metricas["map50"])
    
    if qtd_rodadas_sucesso > 0:
        print(f"\n📊 Calculando a média de {qtd_rodadas_sucesso} rodadas de {dataset_nome}...")
        
        media_map50 = sum(acumulador_metricas["map50"]) / qtd_rodadas_sucesso
        media_map50_95 = sum(acumulador_metricas["map50_95"]) / qtd_rodadas_sucesso
        media_precision = sum(acumulador_metricas["precision"]) / qtd_rodadas_sucesso
        media_recall = sum(acumulador_metricas["recall"]) / qtd_rodadas_sucesso
        
        texto_media = (
            f"{'=' * 40}\n"
            f"MÉDIA FINAL DAS {qtd_rodadas_sucesso} RODADAS - {dataset_nome.upper()}\n"
            f"{'=' * 40}\n"
            f"  mAP50 Média     : {media_map50:.4f}\n"
            f"  mAP50-95 Média  : {media_map50_95:.4f}\n"
            f"  Precision Média : {media_precision:.4f}\n"
            f"  Recall Média    : {media_recall:.4f}\n"
            f"{'=' * 40}\n"
        )
        print(texto_media)
        relatorio_completo_terminal += texto_media

        # Salva a média FINAL dentro da mesma pasta-mãe
        pasta_media = pasta_predict_mae / "media_metricas"
        pasta_media.mkdir(parents=True, exist_ok=True)
        
        caminho_salvar_media = pasta_media / "media_final.txt"
        with open(caminho_salvar_media, "w", encoding="utf-8") as f:
            f.write(texto_media)
            
        print(f"✅ Média salva com sucesso em: {caminho_salvar_media}")
        
        print("\n" + "🌟" * 20)
        print("COPIE O BLOCO ABAIXO E ME MANDE AQUI NO CHAT:")
        print("🌟" * 20 + "\n")
        print(relatorio_completo_terminal)

    else:
        print("\n❌ Nenhuma rodada foi avaliada com sucesso para calcular a média.")

if __name__ == "__main__":
    avaliar_todas_rodadas()