import os
import torch
import gradio as gr
import numpy as np
from cls import ASCClassifier   # Ensure cls.py is in the same directory

# ------------------------------
# Path configuration (adjust as needed)
# ------------------------------
MODEL_DIR = "./models"                # Folder containing .pt files
SPECIES_TXT = "./best_species_kfold.txt"   # File with five species names
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 1. Load species list
# ------------------------------
with open(SPECIES_TXT, 'r') as f:
    species_list = [line.strip() for line in f if line.strip()]
if len(species_list) != 5:
    print(f"Warning: species list length {len(species_list)}, expected 5. Will use first 5 (or pad).")
    if len(species_list) > 5:
        species_list = species_list[:5]
    elif len(species_list) < 5:
        # Pad with placeholders (should not happen in practice)
        species_list += [f"species_{i+1}" for i in range(5 - len(species_list))]
print(f"Key species list: {species_list}")

# ------------------------------
# 2. Load all models
# ------------------------------
model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
model_files.sort()
if not model_files:
    raise RuntimeError(f"No .pt model files found in {MODEL_DIR}")
print(f"Found {len(model_files)} model files")

models = []
model_input_dim = None
transform_clr = True
clr_eps = 1e-6

# Read common parameters and input dimension from the first model
first_ckpt = torch.load(model_files[0], map_location='cpu')
train_args0 = first_ckpt.get('args', {})
transform_clr = train_args0.get('transform_clr', True)
clr_eps = train_args0.get('clr_eps', 1e-6)

# Infer model input dimension (should equal number of species, but safe fallback)
state_dict0 = first_ckpt['model_state_dict']
for key in state_dict0.keys():
    if 'gut_encoder.encoder.0.weight' in key:
        model_input_dim = state_dict0[key].shape[1]
        break
if model_input_dim is None:
    raise ValueError("Cannot infer input dimension from model weights")
print(f"Model expects input dimension: {model_input_dim}")

# Load each model
for path in model_files:
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    train_args = ckpt.get('args', {})

    gut_latent_dim = ckpt.get('gut_dim', train_args.get('gut_latent_dim', 128))
    metadata_latent_dim = train_args.get('metadata_latent_dim', 64)
    classifier_hidden_dims = train_args.get('classifier_hidden_dims', [128, 64, 32])
    dropout = train_args.get('dropout', 0.3)
    use_batch_norm = train_args.get('use_batch_norm', True)
    use_metadata = train_args.get('use_metadata', True)
    metadata_dim = train_args.get('metadata_dim', 0)

    model = ASCClassifier(
        gut_dim=model_input_dim,
        metadata_dim=metadata_dim,
        gut_latent_dim=gut_latent_dim,
        metadata_latent_dim=metadata_latent_dim,
        classifier_hidden_dims=classifier_hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        use_metadata=use_metadata,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
print(f"Successfully loaded {len(models)} models")

# ------------------------------
# 3. Prediction function (receives 5 abundance values)
# ------------------------------
def predict(abundance1, abundance2, abundance3, abundance4, abundance5):
    """
    Takes relative abundances of five key species, returns ASD prediction score (0~1).
    """
    abundances = np.array([abundance1, abundance2, abundance3, abundance4, abundance5], dtype=np.float32)

    # Build model input vector (dimension = model_input_dim)
    raw = np.zeros(model_input_dim, dtype=np.float32)
    # Fill first 5 positions with user input (pad with zeros if dimension >5, truncate if <5)
    n_user = min(len(abundances), model_input_dim)
    raw[:n_user] = abundances[:n_user]

    # Apply arcsinh transform (consistent with training)
    if transform_clr:
        X = np.arcsinh(raw / clr_eps)
    else:
        X = raw

    X_tensor = torch.from_numpy(X.reshape(1, -1)).to(DEVICE)

    # Ensemble prediction
    all_probs = []
    with torch.no_grad():
        for model in models:
            # Get model's metadata dimension (training might have used metadata; zero fill here)
            metadata_dim = getattr(model, 'metadata_dim', 0)
            if metadata_dim > 0:
                x_metadata = torch.zeros(1, metadata_dim, device=DEVICE)
            else:
                x_metadata = torch.zeros(1, 0, device=DEVICE)

            outputs = model(X_tensor, x_metadata)
            prob = outputs["probabilities"].cpu().item()
            all_probs.append(prob)

    avg_prob = np.mean(all_probs).item()
    return avg_prob

# ------------------------------
# 4. Build Gradio interface
# ------------------------------
# Create a numeric input for each species
inputs = [
    gr.Number(label=species_list[0], value=0.0),
    gr.Number(label=species_list[1], value=0.0),
    gr.Number(label=species_list[2], value=0.0),
    gr.Number(label=species_list[3], value=0.0),
    gr.Number(label=species_list[4], value=0.0),
]
outputs = gr.Number(label="ASD Prediction Score", precision=4)

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="ASD Risk Prediction via Gut Microbiota",
    description="Enter the relative abundances of the following five key bacterial species (recommended range 0~1, but you can use your own data range).",
    article="Higher scores (closer to 1) indicate higher ASD risk. This tool uses an ensemble of models (average of multiple models).",
    # allow_flagging="never",   # Uncomment if flagging is not desired
)

# ------------------------------
# 5. Launch the application
# ------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)