from ultralytics import YOLO

def retomar_treinamento():
    print("🚑 Iniciando resgate do treinamento...")
    
    # Aponta para o arquivo last.pt da rodada
    caminho_peso_salvo = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train\flame-yolov8n-run2\weights\last.pt"
    
    caminho_dataset = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\datasets\flame-dataset-new\Detection\dataset.yaml"
    
    model = YOLO(caminho_peso_salvo)
    
    # O resume=True continua aqui
    model.train(resume=True, data=caminho_dataset)
    
    print("✅ Resgate concluído!")

if __name__ == "__main__":
    retomar_treinamento()