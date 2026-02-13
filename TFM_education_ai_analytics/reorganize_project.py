embed_p = base_path / "3_embeddings"
embed_p.mkdir(exist_ok=True)

# 2. Renombrar scripts de modelado
model_path = Path("/workspace/TFM_education_ai_analytics/educational_ai_analytics/modeling")

if (model_path / "train.py").exists():
    print("Renombrando script: train.py -> train_autoencoder.py")
    shutil.move(model_path / "train.py", model_path / "train_autoencoder.py")

if (model_path / "encode.py").exists():
    print("Renombrando script: encode.py -> generate_embeddings.py")
    shutil.move(model_path / "encode.py", model_path / "generate_embeddings.py")

print("✨ Reorganización física completada.")
