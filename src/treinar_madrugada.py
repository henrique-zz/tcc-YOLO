from ultralytics import YOLO
import torch
import time
import os
import gc
import yaml

torch.backends.cudnn.enabled = False

def carregar_config(caminho_yaml="configs/config.yaml"):
    # Lê o seu arquivo YAML existente
    with open(caminho_yaml, 'r', encoding='utf-8') as arquivo:
        return yaml.safe_load(arquivo)

def treinar_lote():
    # Extrai os dados lidos
    config = carregar_config()
    
    # Navega pela hierarquia do YAML (dataset, processamento, treinamento)
    caminho_dataset = os.path.join(config['dataset']['pasta_saida'], "dataset.yaml")
    modelo_base = config['treinamento']['modelo_base']
    epochs = config['treinamento']['epochs']
    batch_size = config['treinamento']['batch_size']
    imgsz = config['treinamento']['imgsz']
    prefixo_run = config['treinamento']['nome_experimento']
    device = config['treinamento']['device']
    workers = config['treinamento']['workers']
    
    # Mantém o caminho absoluto do projeto para evitar erros de diretório
    caminho_projeto = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train"
    
    # Mantém o loop fixo em 10 rodadas para as estatísticas
    for i in range(2, 11):
        nome_pasta = f"{prefixo_run}-{i}"
        
        print(f"\n=======================================================")
        print(f"🚀 INICIANDO {nome_pasta.upper()} ({epochs} ÉPOCAS)")
        print(f"=======================================================\n")
        
        # Carrega os pesos iniciais especificados no YAML
        model = YOLO(modelo_base) 
        
        tempo_inicio = time.time()
        
        model.train(
            data=caminho_dataset,
            epochs=epochs,
            batch=batch_size,          
            imgsz=imgsz,
            project=caminho_projeto,  
            name=nome_pasta,                     
            device=device,
            workers=workers,        
            amp=True,          # Forçado como True para manter a otimização
            cache='disk'       # Forçado no disco para leitura otimizada
        )
        
        tempo_fim = time.time()
        tempo_total_horas = (tempo_fim - tempo_inicio) / 3600
        
        print(f"\n✅ {nome_pasta} finalizada! Levou {tempo_total_horas:.2f} horas.")
        
        # Salvamento automático do tempo
        caminho_pasta_tempo = os.path.join(caminho_projeto, nome_pasta, "tempo")
        os.makedirs(caminho_pasta_tempo, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta_tempo, "tempo_treinamento.txt")
        
        with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(f"Tempo total de treinamento: {tempo_total_horas:.2f} horas\n")

        # Limpeza pesada de memória entre as iterações
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Memória liberada!")
        
        print("⏳ Aguardando 30 segundos antes da próxima run...")
        time.sleep(30)

if __name__ == "__main__":
    treinar_lote()